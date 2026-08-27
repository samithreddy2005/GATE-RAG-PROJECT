"""Tests for the deterministic answer comparison.

This is the most safety-critical function in the project: it decides whether
a student is shown "verified" or "do not trust this". A false positive here
means a wrong derivation is presented as correct, which is exactly the
failure mode the system exists to prevent. Every case below is drawn from a
real answer-key format in data/raw_papers.
"""
from __future__ import annotations

import pytest

from app.gate_generator import answers_match, parse_final_answer


@pytest.mark.parametrize("derived,official,expected", [
    # --- MCQ, single letter -------------------------------------------
    ("B", "B", True),
    ("b", "B", True),
    ("A", "B", False),
    # --- MSQ, order must not matter -----------------------------------
    ("A,B,C", "A,B,C", True),
    ("C,B,A", "A,B,C", True),
    ("A,B", "A,B,C", False),
    ("A,B,C,D", "A,B,C", False),
    # --- NAT ranges, real DA 2024 key formats -------------------------
    ("0.125", "0.125 TO 0.125", True),
    ("0.125", "0.12 TO 0.13", True),
    ("0.131", "0.12 TO 0.13", False),
    ("3", "3 TO 3", True),
    ("2.375", "2.374 TO 2.376", True),
    ("-1.5", "-2 TO -1", True),
    # --- NAT with alternative accepted ranges -------------------------
    ("4", "2 TO 2 OR 4 TO 4", True),
    ("3", "2 TO 2 OR 4 TO 4", False),
    # --- numeric formatting must not cause a false negative -----------
    ("3.0", "3", True),
    ("3.00", "3 TO 3", True),
    # --- empty / missing answers are never a match --------------------
    ("", "B", False),
    ("B", "", False),
    ("", "", False),
])
def test_answers_match(derived, official, expected):
    assert answers_match(derived, official) is expected


def test_non_numeric_derived_against_a_range_is_a_mismatch_not_a_crash():
    """A model that answers "B" to a NAT question must be rejected cleanly."""
    assert answers_match("B", "0.12 TO 0.13") is False


def test_mta_key_never_matches_anything():
    """GATE awards marks to all when a question is annulled. There is no
    correct option, so no derived answer may ever be reported as verified."""
    for candidate in ("A", "B", "C", "D", "0", ""):
        assert answers_match(candidate, "MTA") is False


@pytest.mark.parametrize("text,expected", [
    ("Step 1 ...\nFINAL ANSWER: B", "B"),
    ("final answer: c", "C"),
    ("FINAL ANSWER: B, C, D", "B,C,D"),
    ("FINAL ANSWER: 4096", "4096"),
    ("FINAL ANSWER: **B**", "B"),
    ("FINAL ANSWER: B.", "B"),
    ("no final line here", ""),
])
def test_parse_final_answer(text, expected):
    assert parse_final_answer(text) == expected


def test_parse_final_answer_takes_the_last_occurrence():
    """Models sometimes echo the required format mid-explanation before
    committing to a real answer at the end."""
    text = (
        'I must end with "FINAL ANSWER: X".\n'
        "Step 1: compute the determinant.\n"
        "FINAL ANSWER: C"
    )
    assert parse_final_answer(text) == "C"


def test_a_prose_final_answer_is_treated_as_no_answer():
    """The correction pass explicitly invites the model to dispute the key,
    and it sometimes answers with a sentence. Collapsing that to a
    space-free token produced an unreadable pseudo-answer; it must be
    rejected as "no answer given", which is what it is."""
    text = ("FINAL ANSWER: (the official key is mistaken; the edge can be "
            "either a tree edge or a back edge, so no single option is "
            "uniquely correct)")
    assert parse_final_answer(text) == ""


def test_a_long_but_legitimate_msq_answer_still_parses():
    """The length guard must not reject a real answer."""
    assert parse_final_answer("FINAL ANSWER: A,B,C,D") == "A,B,C,D"
    assert parse_final_answer("FINAL ANSWER: 123456.789") == "123456.789"
