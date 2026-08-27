from __future__ import annotations

import pytest

from app.gate_parser import GateQuestion
from app.question_bank import QuestionBank


@pytest.fixture()
def bank(tmp_path):
    b = QuestionBank(tmp_path / "test.sqlite3")
    b.upsert([
        GateQuestion(subject="DA", year=2024, q_no=17, marks=1, q_type="MCQ",
                     stem="An SVM is trained", official_answer="A",
                     topics=["Machine Learning"], section="DA"),
        GateQuestion(subject="DA", year=2024, q_no=20, marks=2, q_type="MSQ",
                     stem="Which are true of k-means", official_answer="A,C",
                     topics=["Machine Learning", "Data Structures & Algorithms"],
                     section="DA"),
        GateQuestion(subject="DA", year=2024, q_no=13, marks=1, q_type="MCQ",
                     stem="Eigenvalues of M", official_answer="B",
                     topics=["Linear Algebra"], section="DA"),
        GateQuestion(subject="CS", year=2023, q_no=2, marks=1, q_type="MCQ",
                     stem="Annulled question", official_answer="MTA",
                     topics=["General Aptitude"], section="GA"),
    ])
    yield b
    b.close()


def test_filter_by_topic_uses_the_join_table(bank):
    ids = [q.qid for q in bank.filter(topic="Machine Learning")]
    assert ids == ["DA-2024-17", "DA-2024-20"]


def test_filter_by_subject_and_type(bank):
    assert [q.qid for q in bank.filter(subject="DA", q_type="MSQ")] == ["DA-2024-20"]


def test_filter_by_year_range(bank):
    assert [q.qid for q in bank.filter(year_min=2024)] == [
        "DA-2024-13", "DA-2024-17", "DA-2024-20"
    ]


def test_answerable_only_drops_annulled_questions(bank):
    assert [q.qid for q in bank.filter(subject="CS")] == ["CS-2023-2"]
    assert bank.filter(subject="CS", answerable_only=True) == []


def test_upsert_is_idempotent_and_replaces_stale_topics(bank):
    """Re-ingesting after a taxonomy change must not leave the old topic
    attached -- a merge would silently accumulate wrong tags forever."""
    bank.upsert([
        GateQuestion(subject="DA", year=2024, q_no=13, marks=1, q_type="MCQ",
                     stem="Eigenvalues of M", official_answer="B",
                     topics=["Calculus & Optimization"], section="DA"),
    ])
    assert bank.summary()["total_questions"] == 4      # no duplicate row
    assert bank.filter(topic="Linear Algebra") == []   # old tag gone
    assert [q.qid for q in bank.filter(topic="Calculus & Optimization")] == ["DA-2024-13"]


def test_topic_frequency_counts_every_tag(bank):
    freq = bank.topic_frequency(subject="DA")
    assert freq["Machine Learning"] == 2
    assert freq["Linear Algebra"] == 1


def test_topic_marks_weights_by_marks_not_count(bank):
    """Two ML questions worth 1 and 2 marks outweigh one 1-mark LA question,
    which question counts alone would not show."""
    marks = bank.topic_marks(subject="DA")
    assert marks["Machine Learning"] == 3
    assert marks["Linear Algebra"] == 1


def test_get_many_is_a_single_batched_lookup(bank):
    found = bank.get_many(["DA-2024-13", "DA-2024-17", "missing"])
    assert set(found) == {"DA-2024-13", "DA-2024-17"}


def test_sample_is_reproducible_under_a_seed(bank):
    a = [q.qid for q in bank.sample(n=2, seed=42, subject="DA")]
    b = [q.qid for q in bank.sample(n=2, seed=42, subject="DA")]
    assert a == b
    assert len(a) == 2


def test_sample_never_exceeds_the_available_pool(bank):
    assert len(bank.sample(n=99, subject="DA")) == 3


def test_round_trip_preserves_options_and_topics(tmp_path):
    b = QuestionBank(tmp_path / "rt.sqlite3")
    original = GateQuestion(
        subject="DA", year=2024, q_no=1, marks=2, q_type="MCQ",
        stem="stem", options={"A": "alpha", "B": "beta"},
        official_answer="A", topics=["Linear Algebra"], section="DA",
    )
    b.upsert([original])
    restored = b.get("DA-2024-1")
    assert restored == original
    b.close()
