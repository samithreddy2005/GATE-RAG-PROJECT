"""
Builds the searchable corpus from two very different sources and puts them
in one index:

  1. Past papers  -> app.gate_parser -> GateQuestion -> SQLite + one chunk
                     per question (never split mid-question: a stem without
                     its options is unanswerable, so paragraph chunking
                     would actively destroy meaning here).
  2. Concept notes -> app.ingest -> paragraph-aware chunks.

Both land in the same vector store and BM25 index, tagged with
`kind: question | concept`, so one query can surface "the 2024 question that
asked this" *and* "the theorem that answers it".

Run standalone to rebuild everything:

    python -m app.build_index
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.config import (
    CONCEPTS_DIR,
    QUESTION_DB_PATH,
    RAW_PAPERS_DIR,
    UPLOAD_DIR,
)
from app.embeddings import get_embedder
from app.gate_parser import GateQuestion, parse_full_paper
from app.ingest import Chunk, ingest_directory
from app.question_bank import QuestionBank
from app.retriever import BM25Index, HybridRetriever
from app.vectorstore import VectorStore

YEAR_RE = re.compile(r"^\d{4}$")


@dataclass
class Corpus:
    """Everything a request needs, assembled once at startup."""
    bank: QuestionBank
    retriever: HybridRetriever
    vector_store: VectorStore
    bm25_index: BM25Index
    n_questions: int
    n_concept_chunks: int
    papers: list[str]
    problems: list[str] = field(default_factory=list)


def discover_papers(root: Path = RAW_PAPERS_DIR) -> list[tuple[Path, Path, str, int]]:
    """Find (raw_text, answer_key, subject, year) tuples under
    data/raw_papers/<SUBJECT>/<YEAR>/. Directory layout is the contract, so
    adding a paper needs no code change -- only the two files.

    A paper without an answer key is skipped rather than half-ingested:
    without ground truth it cannot be verified, and an unverifiable
    explanation is the one thing this project exists to avoid.
    """
    found = []
    if not root.is_dir():
        return found
    for subject_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for year_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
            if not YEAR_RE.match(year_dir.name):
                continue
            raw = next(iter(sorted(year_dir.glob("*_raw.txt"))), None)
            key = next(iter(sorted(year_dir.glob("*_answer_key.csv"))), None)
            if raw and key:
                found.append((raw, key, subject_dir.name, int(year_dir.name)))
    return found


MCQ_KEY_RE = re.compile(r"^[A-D]$")
MSQ_KEY_RE = re.compile(r"^[A-D](,[A-D])+$")
NAT_KEY_RE = re.compile(r"^-?[\d.]+ TO -?[\d.]+( OR -?[\d.]+ TO -?[\d.]+)*$")


def _compact_ranges(numbers: list[int]) -> str:
    """[16,17,18,20] -> "16-18, 20". Keeps a coverage report readable."""
    if not numbers:
        return ""
    spans: list[tuple[int, int]] = []
    start = previous = numbers[0]
    for n in numbers[1:]:
        if n == previous + 1:
            previous = n
            continue
        spans.append((start, previous))
        start = previous = n
    spans.append((start, previous))
    return ", ".join(f"Q.{a}" if a == b else f"Q.{a}-{b}" for a, b in spans)


def validate_paper(questions: list[GateQuestion], answer_key_csv: Path) -> list[str]:
    """Cheap integrity checks on parsed papers, run at every build.

    These exist because a silently malformed answer key is the worst kind of
    bug in this system: it does not crash, it just marks correct derivations
    as unverified forever. One was found exactly that way during evaluation
    -- CS 2023 Q.60 asks how many rows an SQL query returns (answer: 2) but
    carried the key "2.374 to 2.376", a value that cannot be a row count.
    Every check below is one that would have caught it.
    """
    problems: list[str] = []
    key_rows = {int(r["q_no"]): r for r in
                csv.DictReader(answer_key_csv.open(newline="", encoding="utf-8"))}
    parsed = {q.q_no for q in questions}

    # Coverage is summarized, not listed per question. An incomplete text
    # extraction produces one missing row per absent question, and 50 copies
    # of the same message would bury the integrity problems below that
    # actually need a human to look at them.
    missing = sorted(set(key_rows) - parsed)
    if missing:
        problems.append(
            f"coverage: {len(missing)} of {len(key_rows)} answer-key rows have no "
            f"question text ({_compact_ranges(missing)}) -- the paper extraction "
            f"is incomplete, so these cannot be practised or explained"
        )

    for q in questions:
        if q.q_no not in key_rows:
            problems.append(f"Q.{q.q_no}: question text has no answer key row")
            continue
        if not q.is_answerable:
            continue          # annulled questions are expected to look odd
        key = q.official_answer
        expected = {"MCQ": MCQ_KEY_RE, "MSQ": MSQ_KEY_RE, "NAT": NAT_KEY_RE}.get(q.q_type)
        if expected and not expected.match(key):
            problems.append(
                f"Q.{q.q_no}: {q.q_type} key {key!r} does not look like a valid "
                f"{q.q_type} answer"
            )
        if q.options and MCQ_KEY_RE.match(key) and key not in q.options:
            problems.append(
                f"Q.{q.q_no}: key is ({key}) but the parsed options are "
                f"{sorted(q.options)}"
            )
    return problems


def question_to_chunk(q: GateQuestion) -> Chunk:
    """One chunk per question. Chroma metadata values must be scalars, so
    the topic list is stored as a joined string for display; topic
    *filtering* goes through SQL in QuestionBank, where it belongs."""
    return Chunk(
        id=q.qid,
        text=q.to_chunk_text(),
        source=f"GATE {q.subject} {q.year} Q.{q.q_no}",
        chunk_index=q.q_no,
        metadata={
            "kind": "question",
            "qid": q.qid,
            "subject": q.subject,
            "year": q.year,
            "q_no": q.q_no,
            "q_type": q.q_type,
            "section": q.section,
            "marks": q.marks,
            "topics": ", ".join(q.topics),
            "source": f"GATE {q.subject} {q.year} Q.{q.q_no}",
        },
    )


def build_corpus(rebuild: bool = True) -> Corpus:
    bank = QuestionBank(QUESTION_DB_PATH)

    # ---- 1. structured past papers ------------------------------------
    all_questions: list[GateQuestion] = []
    papers: list[str] = []
    problems: list[str] = []
    for raw_path, key_path, subject, year in discover_papers():
        questions = parse_full_paper(raw_path, key_path, subject, year)
        if not questions:
            continue
        problems += [f"{subject} {year}: {p}"
                     for p in validate_paper(questions, key_path)]
        bank.upsert(questions)
        all_questions.extend(questions)
        papers.append(f"{subject} {year} ({len(questions)} questions)")

    question_chunks = [question_to_chunk(q) for q in all_questions]

    # ---- 2. free-form concept notes + user uploads ---------------------
    concept_chunks = ingest_directory(CONCEPTS_DIR, metadata={"kind": "concept"})
    concept_chunks += ingest_directory(UPLOAD_DIR, metadata={"kind": "concept"})

    chunks = question_chunks + concept_chunks
    if not chunks:
        raise ValueError(
            f"Nothing to index. Add papers under {RAW_PAPERS_DIR} "
            f"or notes under {CONCEPTS_DIR}."
        )

    # ---- 3. embed and index -------------------------------------------
    texts = [c.text for c in chunks]
    embedder = get_embedder()
    embedder.fit(texts)
    embedder.save()
    embeddings = embedder.embed(texts)

    vector_store = VectorStore()
    if rebuild:
        vector_store.reset()
    vector_store.add(
        ids=[c.id for c in chunks],
        texts=texts,
        metadatas=[c.metadata for c in chunks],
        embeddings=embeddings,
    )

    bm25_index = BM25Index()
    bm25_index.fit(chunks)
    bm25_index.save()

    return Corpus(
        bank=bank,
        retriever=HybridRetriever(vector_store, embedder, bm25_index),
        vector_store=vector_store,
        bm25_index=bm25_index,
        n_questions=len(question_chunks),
        n_concept_chunks=len(concept_chunks),
        papers=papers,
        problems=problems,
    )


def main() -> None:
    corpus = build_corpus()
    print(f"Indexed {corpus.n_questions} questions and "
          f"{corpus.n_concept_chunks} concept chunks.")
    for paper in corpus.papers:
        print(f"  - {paper}")
    summary = corpus.bank.summary()
    print(f"Question bank: {summary['total_questions']} rows, "
          f"types {summary['by_type']}")
    top = list(corpus.bank.topic_frequency().items())[:5]
    print("Top topics: " + ", ".join(f"{t} ({n})" for t, n in top))
    if corpus.problems:
        print(f"\nData integrity warnings ({len(corpus.problems)}):")
        for problem in corpus.problems:
            print(f"  ! {problem}")
    else:
        print("\nData integrity: no problems found.")


if __name__ == "__main__":
    main()
