"""
Evaluation harness. Produces the numbers that make this a measurable system
rather than a demo.

Two suites, run independently because they have different costs:

  retrieval  -- offline, deterministic, no API key. Measures whether the
                retriever can find a question from a paraphrase of it, and
                whether it pulls the right concept notes. Recall@k and MRR.

  generation -- calls the LLM once or twice per question, so it costs money
                and time. Measures the headline metric:

                    verified@1  -- derivation matched the official key first try
                    verified@2  -- matched only after the correction retry
                    unverified  -- never matched; shown to the student with a
                                   warning, which is the abstention outcome

The generation numbers are what a reader should judge the project on.
"verified@1" is accuracy that was *checked against ground truth*, not a
self-report -- the comparison never involves a model.

Usage:
    python -m eval.run_eval --retrieval
    python -m eval.run_eval --generation -n 15 --seed 7
    python -m eval.run_eval --all -n 15 --json eval/results.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.build_index import Corpus, build_corpus
from app.config import GROQ_API_KEY
from app.gate_generator import generate_verified_explanation
from app.gate_parser import GateQuestion


# ------------------------------------------------------------------ retrieval

QUERY_SET_PATH = Path(__file__).resolve().parent / "student_queries.json"


@dataclass
class RetrievalResult:
    n_queries: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    concept_precision: float
    abstention_rate: float
    misses: list[str] = field(default_factory=list)
    false_answers: list[str] = field(default_factory=list)


def _echo_query(q: GateQuestion) -> str:
    """Sanity-check query only: the question's own topic terms plus the
    opening of its stem. It is reported separately and should NOT be read as
    a retrieval score -- it largely measures string overlap, and it scores
    near 100% for any working index. Its only job is to catch a corpus that
    failed to index at all."""
    first_line = q.stem.strip().split("\n")[0]
    return f"{' '.join(q.topics)} {' '.join(first_line.split()[:12])}"


def evaluate_echo_sanity(corpus: Corpus, k: int = 5) -> float:
    questions = corpus.bank.filter(answerable_only=True)
    found = 0
    for q in questions:
        hits = corpus.retriever.retrieve(_echo_query(q), top_k=k)
        if q.qid in [h.metadata.get("qid") for h in hits if h.kind == "question"]:
            found += 1
    return round(found / (len(questions) or 1), 3)


def evaluate_retrieval(corpus: Corpus, k: int = 5) -> RetrievalResult:
    """The real benchmark: hand-authored, student-phrased queries that share
    minimal wording with the question stems, plus off-topic queries the
    system is required to refuse."""
    spec = json.loads(QUERY_SET_PATH.read_text(encoding="utf-8"))
    ranks: list[int | None] = []
    concept_ok = concept_total = 0
    misses: list[str] = []

    for case in spec["queries"]:
        hits = corpus.retriever.retrieve(case["query"], top_k=k)
        if case["expect_qid"]:
            ids = [h.metadata.get("qid") for h in hits if h.kind == "question"]
            rank = ids.index(case["expect_qid"]) + 1 if case["expect_qid"] in ids else None
            ranks.append(rank)
            if rank is None:
                misses.append(f"{case['expect_qid']} <- {case['query'][:45]}")
        if case["expect_kind"] in ("concept", "both"):
            concept_total += 1
            if any(h.kind == "concept" for h in hits):
                concept_ok += 1

    # Off-topic queries must return nothing. A retrieval system that always
    # returns its top-k is not "high recall", it is incapable of saying no --
    # and every one of those results becomes grounding for a confident,
    # wrong answer.
    false_answers = []
    for query in spec["off_topic"]:
        if corpus.retriever.retrieve(query, top_k=k):
            false_answers.append(query)
    off_topic_total = len(spec["off_topic"]) or 1

    n = len(ranks) or 1
    return RetrievalResult(
        n_queries=len(spec["queries"]),
        recall_at_1=round(sum(1 for r in ranks if r == 1) / n, 3),
        recall_at_3=round(sum(1 for r in ranks if r and r <= 3) / n, 3),
        recall_at_5=round(sum(1 for r in ranks if r and r <= 5) / n, 3),
        mrr=round(sum(1 / r for r in ranks if r) / n, 3),
        concept_precision=round(concept_ok / (concept_total or 1), 3),
        abstention_rate=round((off_topic_total - len(false_answers)) / off_topic_total, 3),
        misses=misses,
        false_answers=false_answers,
    )


# ----------------------------------------------------------------- generation

@dataclass
class GenerationResult:
    n: int
    verified_at_1: float
    verified_at_2: float
    unverified: float
    truncated: int
    errors: int
    median_seconds: float
    by_type: dict[str, dict] = field(default_factory=dict)
    failures: list[dict] = field(default_factory=list)


def evaluate_generation(corpus: Corpus, n: int, seed: int | None) -> GenerationResult:
    questions = corpus.bank.sample(n=n, seed=seed, answerable_only=True)
    counts = {"verified": 0, "corrected": 0, "unverified": 0, "truncated": 0, "error": 0}
    per_type: dict[str, dict] = {}
    durations: list[float] = []
    failures: list[dict] = []

    for i, q in enumerate(questions, 1):
        started = time.perf_counter()
        chunks = corpus.retriever.retrieve(
            f"{' '.join(q.topics)} {q.stem}", top_k=4, where={"kind": "concept"}
        )
        explanation = generate_verified_explanation(q, chunks)
        elapsed = time.perf_counter() - started
        durations.append(elapsed)

        counts[explanation.status] = counts.get(explanation.status, 0) + 1
        bucket = per_type.setdefault(q.q_type, {"n": 0, "verified": 0})
        bucket["n"] += 1
        if explanation.verified:
            bucket["verified"] += 1

        if not explanation.verified:
            failures.append({
                "qid": q.qid, "q_type": q.q_type, "topics": q.topics,
                "derived": explanation.derived_answer,
                "official": explanation.official_answer,
                "status": explanation.status,
            })

        flag = {"verified": "ok", "corrected": "ok (retry)",
                "unverified": "FAILED", "truncated": "TRUNCATED",
                "error": "ERROR"}[explanation.status]
        print(f"  [{i}/{len(questions)}] {q.qid:<12} {q.q_type:<4} "
              f"derived={explanation.derived_answer or '-':<8} "
              f"official={explanation.official_answer:<14} {flag}  ({elapsed:.1f}s)")

    total = len(questions) or 1
    for bucket in per_type.values():
        bucket["verified_rate"] = round(bucket["verified"] / bucket["n"], 3)

    return GenerationResult(
        n=len(questions),
        verified_at_1=round(counts["verified"] / total, 3),
        verified_at_2=round(counts["corrected"] / total, 3),
        unverified=round(counts["unverified"] / total, 3),
        truncated=counts["truncated"],
        errors=counts["error"],
        median_seconds=round(statistics.median(durations), 2) if durations else 0.0,
        by_type=per_type,
        failures=failures,
    )


# ----------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description="GATE RAG evaluation harness")
    parser.add_argument("--retrieval", action="store_true", help="run the offline retrieval suite")
    parser.add_argument("--generation", action="store_true", help="run the LLM suite (needs GROQ_API_KEY)")
    parser.add_argument("--all", action="store_true", help="run both suites")
    parser.add_argument("-n", type=int, default=10, help="questions to sample for generation")
    parser.add_argument("--seed", type=int, default=42, help="sampling seed, for reproducibility")
    parser.add_argument("--json", type=Path, help="write results to this file")
    args = parser.parse_args()

    if not (args.retrieval or args.generation or args.all):
        args.retrieval = True   # the suite that always runs without a key

    print("Building index...")
    corpus = build_corpus()
    print(f"  {corpus.n_questions} questions, {corpus.n_concept_chunks} concept chunks\n")

    results: dict = {}

    if args.retrieval or args.all:
        print("== Retrieval (hand-authored student-phrased queries) ==")
        r = evaluate_retrieval(corpus)
        results["retrieval"] = asdict(r)
        print(f"  queries            {r.n_queries}")
        print(f"  recall@1           {r.recall_at_1:.1%}")
        print(f"  recall@3           {r.recall_at_3:.1%}")
        print(f"  recall@5           {r.recall_at_5:.1%}")
        print(f"  MRR                {r.mrr:.3f}")
        print(f"  concept precision  {r.concept_precision:.1%}   "
              f"(query surfaced the explanatory notes it should)")
        print(f"  abstention rate    {r.abstention_rate:.1%}   "
              f"(off-topic queries correctly refused)")
        if r.misses:
            print("  missed:")
            for miss in r.misses:
                print(f"    - {miss}")
        if r.false_answers:
            print("  WRONGLY ANSWERED off-topic queries:")
            for query in r.false_answers:
                print(f"    - {query}")
        echo = evaluate_echo_sanity(corpus)
        results["echo_sanity"] = echo
        print(f"  [sanity] echo-query recall@5 {echo:.1%}  "
              f"(near-duplicate queries; not a retrieval score)")
        print()

    if args.generation or args.all:
        print("== Generation (verified against the official answer key) ==")
        if not GROQ_API_KEY:
            print("  SKIPPED: no GROQ_API_KEY set in rag-project/.env\n")
        else:
            g = evaluate_generation(corpus, n=args.n, seed=args.seed)
            results["generation"] = asdict(g)
            print()
            print(f"  sampled          {g.n}")
            print(f"  verified@1       {g.verified_at_1:.1%}   (correct on the first derivation)")
            print(f"  verified@2       {g.verified_at_2:.1%}   (correct only after the retry)")
            print(f"  total verified   {g.verified_at_1 + g.verified_at_2:.1%}")
            print(f"  unverified       {g.unverified:.1%}   (shown to the student with a warning)")
            print(f"  truncated        {g.truncated}   (token limit hit before an answer)")
            print(f"  errors           {g.errors}")
            print(f"  median latency   {g.median_seconds}s")
            for q_type, bucket in sorted(g.by_type.items()):
                print(f"    {q_type:<4} {bucket['verified']}/{bucket['n']} "
                      f"verified ({bucket['verified_rate']:.0%})")
            print()

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Wrote {args.json}")

    corpus.bank.close()


if __name__ == "__main__":
    main()
