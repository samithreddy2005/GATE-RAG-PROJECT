"""API tests via FastAPI's TestClient. Generation endpoints are not exercised
here -- they need a live Groq key and would make the suite non-deterministic.
Everything up to the model call is covered."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app, _safe_filename


@pytest.fixture(scope="module")
def client():
    # The context manager triggers lifespan, which builds the index.
    with TestClient(app) as c:
        yield c


def test_status_reports_a_ready_index(client):
    body = client.get("/api/status").json()
    assert body["ready"] is True
    assert body["indexed_questions"] > 0
    assert body["bank"]["total_questions"] > 0


def test_subjects_lists_both_papers(client):
    assert {"DA", "CS"} <= set(client.get("/api/subjects").json()["subjects"])


def test_topics_returns_marks_weighted_shares(client):
    topics = client.get("/api/topics", params={"subject": "DA"}).json()["topics"]
    assert topics
    assert all({"topic", "questions", "marks", "marks_share"} <= set(t) for t in topics)
    # Sorted by question count descending.
    counts = [t["questions"] for t in topics]
    assert counts == sorted(counts, reverse=True)


def test_questions_filter_by_topic(client):
    body = client.get("/api/questions", params={"topic": "General Aptitude"}).json()
    assert body["count"] > 0
    assert all("General Aptitude" in q["topics"] for q in body["questions"])


def test_question_detail_404s_on_unknown_id(client):
    assert client.get("/api/questions/DA-1999-1").status_code == 404


def test_practice_withholds_the_official_answer(client):
    """The key must not reach the browser before the student answers --
    otherwise practice mode is trivially cheatable from devtools."""
    body = client.get("/api/practice", params={"n": 3, "seed": 1}).json()
    assert body["count"] == 3
    for q in body["questions"]:
        assert "official_answer" not in q
        assert q["stem"]


def test_practice_is_reproducible_under_a_seed(client):
    a = client.get("/api/practice", params={"n": 3, "seed": 7}).json()
    b = client.get("/api/practice", params={"n": 3, "seed": 7}).json()
    assert [q["qid"] for q in a["questions"]] == [q["qid"] for q in b["questions"]]


def test_attempt_grades_a_correct_answer(client):
    q = client.get("/api/questions", params={"q_type": "MCQ", "limit": 1}).json()["questions"][0]
    body = client.post("/api/attempt", json={"qid": q["qid"], "answer": q["official_answer"]}).json()
    assert body["correct"] is True
    assert body["marks_awarded"] == float(q["marks"])


def test_attempt_applies_mcq_negative_marking(client):
    q = client.get("/api/questions", params={"q_type": "MCQ", "limit": 1}).json()["questions"][0]
    wrong = "D" if q["official_answer"] != "D" else "A"
    body = client.post("/api/attempt", json={"qid": q["qid"], "answer": wrong}).json()
    assert body["correct"] is False
    assert body["marks_awarded"] < 0


def test_attempt_does_not_negatively_mark_nat(client):
    """GATE applies negative marking to MCQ only."""
    nat = client.get("/api/questions", params={"q_type": "NAT", "limit": 1}).json()["questions"]
    if not nat:
        pytest.skip("no NAT questions in the corpus")
    body = client.post("/api/attempt", json={"qid": nat[0]["qid"], "answer": "-999"}).json()
    assert body["correct"] is False
    assert body["marks_awarded"] == 0.0


def test_attempt_404s_on_unknown_question(client):
    assert client.post("/api/attempt", json={"qid": "XX-1900-1", "answer": "A"}).status_code == 404


def test_chat_streams_sources_before_tokens(client):
    with client.stream("POST", "/api/chat",
                       json={"question": "eigenvalues of a 2x2 matrix"}) as r:
        assert r.status_code == 200
        first = next(line for line in r.iter_lines() if line.startswith("data: "))
    assert '"type": "sources"' in first


def test_chat_rejects_an_empty_question(client):
    assert client.post("/api/chat", json={"question": ""}).status_code == 422


@pytest.mark.parametrize("raw,expected", [
    ("notes.md", "notes.md"),
    ("../../etc/passwd", "passwd"),
    ("..\\..\\windows\\system32\\evil.txt", "evil.txt"),
    ("/absolute/path/notes.txt", "notes.txt"),
    ("we ird name!.md", "we_ird_name_.md"),
    ("", "upload.txt"),
    ("...", "upload.txt"),
])
def test_upload_filenames_are_sanitized(raw, expected):
    """Uploads are written to disk, so a client-supplied name must never be
    able to escape the upload directory."""
    assert _safe_filename(raw) == expected


def test_ingest_rejects_unsupported_file_types(client):
    r = client.post("/api/ingest", files={"files": ("evil.exe", b"MZ", "application/octet-stream")})
    assert r.status_code == 400
