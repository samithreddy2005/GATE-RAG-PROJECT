"""Tests for the answer-key integrity checks.

A malformed answer key is the most dangerous bug class in this project: it
does not crash, it just marks correct derivations as unverified forever. The
CS 2023 Q.60 case below is real -- it was found by the evaluation harness,
not by a test, which is why these checks now exist.
"""
from __future__ import annotations

import textwrap

import pytest

from app.build_index import _compact_ranges, validate_paper
from app.gate_parser import GateQuestion


def write_key(tmp_path, rows: str):
    path = tmp_path / "key.csv"
    path.write_text(textwrap.dedent(rows).strip() + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("numbers,expected", [
    ([], ""),
    ([5], "Q.5"),
    ([1, 2, 3], "Q.1-3"),
    ([1, 3], "Q.1, Q.3"),
    ([16, 17, 18, 20, 30, 31], "Q.16-18, Q.20, Q.30-31"),
])
def test_compact_ranges(numbers, expected):
    assert _compact_ranges(numbers) == expected


def test_nat_key_that_cannot_be_a_row_count_is_flagged(tmp_path):
    """The real CS 2023 Q.60 defect: the question asks how many rows an SQL
    query returns, and the key said "2.374 to 2.376"."""
    key = write_key(tmp_path, """
        q_no,type,subject,key,marks
        60,NAT,CS,"2.374 to 2.376",2
    """)
    q = GateQuestion(subject="CS", year=2023, q_no=60, marks=2, q_type="MCQ",
                     stem="The number of rows returned is ____",
                     options={"A": "1", "B": "2"}, official_answer="2.374 TO 2.376")
    problems = validate_paper([q], key)
    assert any("does not look like a valid MCQ answer" in p for p in problems)


def test_valid_keys_produce_no_problems(tmp_path):
    key = write_key(tmp_path, """
        q_no,type,subject,key,marks
        1,MCQ,DA,B,1
        2,MSQ,DA,"A,C",2
        3,NAT,DA,"0.12 to 0.13",2
    """)
    questions = [
        GateQuestion(subject="DA", year=2024, q_no=1, marks=1, q_type="MCQ",
                     stem="x", options={"A": "a", "B": "b"}, official_answer="B"),
        GateQuestion(subject="DA", year=2024, q_no=2, marks=2, q_type="MSQ",
                     stem="x", options={"A": "a", "C": "c"}, official_answer="A,C"),
        GateQuestion(subject="DA", year=2024, q_no=3, marks=2, q_type="NAT",
                     stem="x", official_answer="0.12 TO 0.13"),
    ]
    assert validate_paper(questions, key) == []


def test_key_pointing_at_a_missing_option_is_flagged(tmp_path):
    """Key says (D) but the parser only recovered options A-C, which means
    the extraction dropped a line."""
    key = write_key(tmp_path, """
        q_no,type,subject,key,marks
        1,MCQ,DA,D,1
    """)
    q = GateQuestion(subject="DA", year=2024, q_no=1, marks=1, q_type="MCQ",
                     stem="x", options={"A": "a", "B": "b", "C": "c"},
                     official_answer="D")
    problems = validate_paper([q], key)
    assert any("parsed options" in p for p in problems)


def test_missing_question_text_is_reported_as_one_coverage_line(tmp_path):
    key = write_key(tmp_path, """
        q_no,type,subject,key,marks
        1,MCQ,DA,A,1
        2,MCQ,DA,B,1
        3,MCQ,DA,C,1
    """)
    q = GateQuestion(subject="DA", year=2024, q_no=1, marks=1, q_type="MCQ",
                     stem="x", options={"A": "a"}, official_answer="A")
    problems = validate_paper([q], key)
    coverage = [p for p in problems if p.startswith("coverage:")]
    assert len(coverage) == 1
    assert "2 of 3" in coverage[0]
    assert "Q.2-3" in coverage[0]


def test_annulled_questions_are_not_flagged_as_malformed(tmp_path):
    """An "MTA" key is a legitimate GATE outcome, not a data error."""
    key = write_key(tmp_path, """
        q_no,type,subject,key,marks
        1,MCQ,CS,MTA,1
    """)
    q = GateQuestion(subject="CS", year=2023, q_no=1, marks=1, q_type="MCQ",
                     stem="x", options={"A": "a"}, official_answer="MTA")
    assert validate_paper([q], key) == []


def test_the_shipped_corpus_has_no_key_integrity_problems():
    """Coverage gaps are expected in this sample corpus; malformed keys are
    not. This test would have failed before CS 2023 Q.60 was corrected."""
    from app.build_index import discover_papers
    from app.gate_parser import parse_full_paper

    for raw, key, subject, year in discover_papers():
        questions = parse_full_paper(raw, key, subject, year)
        problems = [p for p in validate_paper(questions, key)
                    if not p.startswith("coverage:")]
        assert problems == [], f"{subject} {year}: {problems}"
