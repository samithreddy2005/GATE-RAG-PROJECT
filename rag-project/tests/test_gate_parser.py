"""Parser tests run against the real official papers in data/raw_papers,
not synthetic fixtures. Every messy case asserted here was found in an
actual GATE PDF text extraction."""
from __future__ import annotations

import pytest

from app.gate_parser import (
    GateQuestion,
    normalize_answer_key,
    parse_paper,
    tag_topics,
)

RAW_WITH_NOISE = """\
Data Science and Artificial Intelligence (DA)
Page 1 of 41
Organizing Institute: IISc Bengaluru
Q.1 - Q.5 Carry ONE mark Each
Q.1 Which ONE of the following is a measure of central tendency?
(A) variance
(B) mean
(C) range
(D) skewness
Q.6 - Q.10 Carry TWO marks Each
Q.6 The value of the determinant of the 2x2 identity matrix is ______.
"""


def test_marks_banner_is_not_parsed_as_a_question():
    """"Q.6 - Q.10 Carry TWO marks Each" starts with "Q.6" and would be
    picked up as question 6 by a naive regex. Question 6 must come from the
    real question text below it, not the banner."""
    questions = parse_paper(RAW_WITH_NOISE, subject="DA", year=2024)
    q_nos = [q.q_no for q in questions]
    assert q_nos == [1, 6]
    q6 = questions[1]
    assert "determinant" in q6.stem
    assert "Carry TWO marks" not in q6.stem


def test_marks_banner_assigns_marks_to_a_range():
    questions = {q.q_no: q for q in parse_paper(RAW_WITH_NOISE, "DA", 2024)}
    assert questions[1].marks == 1
    assert questions[6].marks == 2


def test_page_headers_are_stripped_from_stems():
    questions = parse_paper(RAW_WITH_NOISE, "DA", 2024)
    joined = " ".join(q.stem for q in questions)
    assert "Page 1 of 41" not in joined
    assert "Organizing Institute" not in joined


def test_mcq_options_are_extracted_and_nat_has_none():
    questions = {q.q_no: q for q in parse_paper(RAW_WITH_NOISE, "DA", 2024)}
    assert questions[1].q_type == "MCQ"
    assert set(questions[1].options) == {"A", "B", "C", "D"}
    assert questions[1].options["B"] == "mean"
    assert questions[6].q_type == "NAT"
    assert questions[6].options == {}


def test_duplicate_question_number_keeps_the_more_complete_block():
    """Page breaks in the real PDFs repeat a question marker, leaving one
    truncated copy and one complete copy."""
    raw = (
        "Q.4 The sum of the series is\n"
        "Q.4 The sum of the series 1 + 1/2 + 1/4 + ... is\n"
        "(A) 1\n(B) 2\n(C) 3\n(D) 4\n"
    )
    questions = parse_paper(raw, "DA", 2024)
    assert len(questions) == 1
    assert questions[0].options  # the copy with options won


@pytest.mark.parametrize("raw,expected", [
    ("A", "A"),
    ("a", "A"),
    ("A;B", "A,B"),          # DA key delimiter
    ("A,B", "A,B"),          # CS key delimiter
    ("A; B ; C", "A,B,C"),
    (" 0.12 to 0.13 ", "0.12 TO 0.13"),
    ("MTA", "MTA"),
])
def test_answer_key_normalization(raw, expected):
    assert normalize_answer_key(raw) == expected


def test_general_aptitude_section_bypasses_the_subject_taxonomy():
    """GA questions share no vocabulary with the DA syllabus. The key labels
    them "GA", and that label must win over keyword matching -- otherwise a
    verbal-reasoning question lands in Uncategorized or, worse, a wrong
    technical topic."""
    q = GateQuestion(subject="DA", year=2024, q_no=1, marks=1, q_type="MCQ",
                     stem="The meaning of the word is analogous to", section="GA")
    tag_topics([q])
    assert q.topics == ["General Aptitude"]


def test_word_boundary_matching_avoids_the_mean_in_meaning_false_positive():
    """A real bug found during development: substring matching tagged a
    vocabulary question as Probability & Statistics because "mean" occurs
    inside "meaning"."""
    q = GateQuestion(subject="DA", year=2024, q_no=2, marks=1, q_type="MCQ",
                     stem="What is the meaning of the given word?", section="DA")
    tag_topics([q])
    assert "Probability & Statistics" not in q.topics


def test_hyphen_and_space_are_interchangeable_in_keywords():
    """Real GATE phrasing uses "decision-tree" where the taxonomy lists
    "decision tree"."""
    q = GateQuestion(subject="DA", year=2024, q_no=3, marks=2, q_type="MCQ",
                     stem="A decision-tree classifier is trained on the data.",
                     section="DA")
    tag_topics([q])
    assert "Machine Learning" in q.topics


def test_answerable_flag_excludes_dropped_questions():
    dropped = GateQuestion(subject="CS", year=2023, q_no=2, marks=1, q_type="MCQ",
                           stem="x", official_answer="MTA")
    normal = GateQuestion(subject="CS", year=2023, q_no=3, marks=1, q_type="MCQ",
                          stem="x", official_answer="A")
    assert not dropped.is_answerable
    assert normal.is_answerable


def test_qid_is_stable_and_unique_per_paper():
    q = GateQuestion(subject="DA", year=2024, q_no=17, marks=2, q_type="MCQ", stem="x")
    assert q.qid == "DA-2024-17"


def test_chunk_text_includes_options_and_provenance():
    q = GateQuestion(subject="DA", year=2024, q_no=5, marks=1, q_type="MCQ",
                     stem="Pick one", options={"A": "alpha", "B": "beta"},
                     topics=["Linear Algebra"])
    text = q.to_chunk_text()
    assert "GATE DA 2024 Q.5" in text
    assert "Linear Algebra" in text
    assert "(A) alpha" in text and "(B) beta" in text
