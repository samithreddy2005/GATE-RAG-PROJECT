"""End-to-end retrieval tests against the real corpus.

These are the tests that would have caught the project's original defect:
the GATE modules existed but nothing indexed them, so no query could ever
reach a past question.
"""
from __future__ import annotations

import pytest

from app.build_index import build_corpus, discover_papers
from app.ingest import chunk_text


@pytest.fixture(scope="module")
def corpus():
    return build_corpus(rebuild=True)


def test_papers_are_discovered_from_the_directory_layout():
    found = discover_papers()
    assert found, "no papers discovered under data/raw_papers"
    subjects = {subject for _, _, subject, _ in found}
    assert {"DA", "CS"} <= subjects


def test_both_questions_and_concepts_are_indexed(corpus):
    assert corpus.n_questions > 0
    assert corpus.n_concept_chunks > 0
    assert corpus.vector_store.count() == corpus.n_questions + corpus.n_concept_chunks


def test_retrieval_finds_the_matching_past_question(corpus):
    hits = corpus.retriever.retrieve(
        "probability of two girls and one boy in a family of three children", top_k=5
    )
    assert any(h.kind == "question" for h in hits)
    assert any("DA 2024" in h.source for h in hits)


def test_retrieval_surfaces_the_concept_that_explains_a_question(corpus):
    """The point of mixing both kinds in one index: the exam question AND
    the theorem behind it should come back together."""
    hits = corpus.retriever.retrieve(
        "eigenvalues of a 2x2 matrix complex conjugate pair", top_k=5
    )
    kinds = {h.kind for h in hits}
    assert "concept" in kinds and "question" in kinds


def test_metadata_filter_restricts_results(corpus):
    hits = corpus.retriever.retrieve("sorting complexity", top_k=8,
                                     where={"kind": "concept"})
    assert hits
    assert all(h.kind == "concept" for h in hits)


def test_subject_filter_restricts_to_one_paper(corpus):
    hits = corpus.retriever.retrieve("algorithm", top_k=8, where={"subject": "DA"})
    assert all(h.metadata.get("subject") == "DA" for h in hits)


def test_results_are_ordered_by_descending_score(corpus):
    hits = corpus.retriever.retrieve("bayes theorem conditional probability", top_k=5)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_nonsense_query_abstains_rather_than_returning_junk(corpus):
    """Below CONFIDENCE_THRESHOLD the retriever returns nothing, so the
    generator tells the student it does not know instead of improvising."""
    hits = corpus.retriever.retrieve("zzzqqqx nonexistent gibberish token", top_k=5)
    assert hits == []


def test_retrieved_questions_resolve_back_to_the_bank(corpus):
    """Retrieval ids must be usable as question-bank keys -- that link is
    what lets a search result become an explainable question."""
    hits = corpus.retriever.retrieve("machine learning classifier", top_k=8)
    qids = [h.metadata["qid"] for h in hits if h.kind == "question"]
    assert qids
    assert set(corpus.bank.get_many(qids)) == set(qids)


def test_chunk_ids_are_content_addressed_and_stable():
    """Re-indexing an unchanged file must produce the same ids, otherwise
    every rebuild duplicates the corpus."""
    text = "Para one.\n\nPara two.\n\nPara three."
    first = chunk_text(text, source="notes.md")
    second = chunk_text(text, source="notes.md")
    assert [c.id for c in first] == [c.id for c in second]


def test_changed_content_produces_different_chunk_ids():
    a = chunk_text("Original text.", source="notes.md")
    b = chunk_text("Edited text.", source="notes.md")
    assert a[0].id != b[0].id
