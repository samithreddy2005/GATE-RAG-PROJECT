"""
Structured question bank: SQLite holds the ground-truth structured records
(subject, year, topic, official answer), while the embeddings/BM25
machinery (app.embeddings, app.retriever) indexes the flattened question
text for semantic + keyword search.

This split matters. Metadata filters (year range, topic, subject) run as a
fast SQL WHERE clause, and only the *remaining* candidates go through
similarity search. That is both faster and more precise than trying to
encode "year > 2020" into a vector -- embeddings have no reliable notion
of numeric ordering.
"""
from __future__ import annotations

import json
import random
import sqlite3
import threading
from pathlib import Path

from app.gate_parser import GateQuestion

SCHEMA = """
CREATE TABLE IF NOT EXISTS questions (
    id              TEXT PRIMARY KEY,
    subject         TEXT NOT NULL,
    year            INTEGER NOT NULL,
    q_no            INTEGER NOT NULL,
    marks           INTEGER NOT NULL,
    q_type          TEXT NOT NULL,
    section         TEXT NOT NULL DEFAULT '',
    stem            TEXT NOT NULL,
    options_json    TEXT NOT NULL,
    official_answer TEXT NOT NULL,
    topics_json     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_subject_year ON questions(subject, year);
CREATE INDEX IF NOT EXISTS idx_q_type ON questions(q_type);

-- Normalized topic table. Topics live in their own rows (not only inside
-- topics_json) so "give me every Machine Learning question from 2024" is a
-- single indexed JOIN instead of loading every row and filtering in Python.
CREATE TABLE IF NOT EXISTS question_topics (
    question_id TEXT NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    topic       TEXT NOT NULL,
    PRIMARY KEY (question_id, topic)
);
CREATE INDEX IF NOT EXISTS idx_topic ON question_topics(topic);
"""


class QuestionBank:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: FastAPI runs sync endpoint handlers in a
        # threadpool, so the connection is touched from several threads. The
        # lock below serializes access, which is correct and more than fast
        # enough at question-bank scale (thousands of rows, not millions).
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- write

    def upsert(self, questions: list[GateQuestion]) -> list[str]:
        ids = []
        with self._lock, self.conn:
            for q in questions:
                qid = q.qid
                ids.append(qid)
                self.conn.execute(
                    """INSERT INTO questions
                       (id, subject, year, q_no, marks, q_type, section, stem,
                        options_json, official_answer, topics_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(id) DO UPDATE SET
                         marks=excluded.marks, q_type=excluded.q_type,
                         section=excluded.section,
                         stem=excluded.stem, options_json=excluded.options_json,
                         official_answer=excluded.official_answer,
                         topics_json=excluded.topics_json""",
                    (qid, q.subject, q.year, q.q_no, q.marks, q.q_type, q.section, q.stem,
                     json.dumps(q.options), q.official_answer, json.dumps(q.topics)),
                )
                # Replace rather than merge: re-ingesting a paper after a
                # taxonomy change must not leave stale topics behind.
                self.conn.execute("DELETE FROM question_topics WHERE question_id=?", (qid,))
                self.conn.executemany(
                    "INSERT OR IGNORE INTO question_topics (question_id, topic) VALUES (?, ?)",
                    [(qid, t) for t in q.topics],
                )
        return ids

    # ----------------------------------------------------------------- read

    def get(self, qid: str) -> GateQuestion | None:
        with self._lock:
            row = self.conn.execute("SELECT * FROM questions WHERE id=?", (qid,)).fetchone()
        return self._row_to_question(row) if row else None

    def get_many(self, qids: list[str]) -> dict[str, GateQuestion]:
        """Batch fetch, so resolving a page of retrieval hits is one query."""
        if not qids:
            return {}
        placeholders = ",".join("?" * len(qids))
        with self._lock:
            rows = self.conn.execute(
                f"SELECT * FROM questions WHERE id IN ({placeholders})", qids
            ).fetchall()
        return {r["id"]: self._row_to_question(r) for r in rows}

    def filter(self, subject: str | None = None, year_min: int | None = None,
               year_max: int | None = None, topic: str | None = None,
               q_type: str | None = None, answerable_only: bool = False,
               limit: int | None = None) -> list[GateQuestion]:
        sql = "SELECT q.* FROM questions q"
        clauses: list[str] = []
        params: list = []
        if topic:
            sql += " JOIN question_topics t ON t.question_id = q.id"
            clauses.append("t.topic = ?")
            params.append(topic)
        if subject:
            clauses.append("q.subject = ?")
            params.append(subject)
        if year_min is not None:
            clauses.append("q.year >= ?")
            params.append(year_min)
        if year_max is not None:
            clauses.append("q.year <= ?")
            params.append(year_max)
        if q_type:
            clauses.append("q.q_type = ?")
            params.append(q_type)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY q.subject, q.year, q.q_no"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)

        with self._lock:
            rows = self.conn.execute(sql, params).fetchall()
        questions = [self._row_to_question(r) for r in rows]
        if answerable_only:
            questions = [q for q in questions if q.is_answerable]
        return questions

    def sample(self, n: int = 1, seed: int | None = None, **filters) -> list[GateQuestion]:
        """Random practice set. Sampling happens in Python over the filtered
        list rather than via SQL ORDER BY RANDOM() so that passing `seed`
        makes an evaluation run exactly reproducible."""
        pool = self.filter(**filters)
        rng = random.Random(seed)
        return rng.sample(pool, min(n, len(pool)))

    # ------------------------------------------------------------ analytics

    def topic_frequency(self, subject: str | None = None,
                        year_min: int | None = None,
                        year_max: int | None = None) -> dict[str, int]:
        """Powers the "which topics show up most" view -- the answer a
        student actually wants when deciding what to revise next."""
        sql = ("SELECT t.topic AS topic, COUNT(*) AS n "
               "FROM question_topics t JOIN questions q ON q.id = t.question_id")
        clauses, params = [], []
        if subject:
            clauses.append("q.subject = ?")
            params.append(subject)
        if year_min is not None:
            clauses.append("q.year >= ?")
            params.append(year_min)
        if year_max is not None:
            clauses.append("q.year <= ?")
            params.append(year_max)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " GROUP BY t.topic ORDER BY n DESC, t.topic ASC"
        with self._lock:
            rows = self.conn.execute(sql, params).fetchall()
        return {r["topic"]: r["n"] for r in rows}

    def topic_marks(self, subject: str | None = None) -> dict[str, int]:
        """Frequency counts questions; this weights by marks. A topic with
        three 2-mark questions is worth more to a candidate score than one
        with four 1-mark questions, and only this view surfaces that."""
        sql = ("SELECT t.topic AS topic, SUM(q.marks) AS m "
               "FROM question_topics t JOIN questions q ON q.id = t.question_id")
        params: list = []
        if subject:
            sql += " WHERE q.subject = ?"
            params.append(subject)
        sql += " GROUP BY t.topic ORDER BY m DESC, t.topic ASC"
        with self._lock:
            rows = self.conn.execute(sql, params).fetchall()
        return {r["topic"]: int(r["m"]) for r in rows}

    def summary(self) -> dict:
        with self._lock:
            total = self.conn.execute("SELECT COUNT(*) AS n FROM questions").fetchone()["n"]
            papers = self.conn.execute(
                "SELECT subject, year, COUNT(*) AS n, SUM(marks) AS marks "
                "FROM questions GROUP BY subject, year ORDER BY subject, year"
            ).fetchall()
            types = self.conn.execute(
                "SELECT q_type, COUNT(*) AS n FROM questions GROUP BY q_type ORDER BY n DESC"
            ).fetchall()
        return {
            "total_questions": total,
            "papers": [
                {"subject": r["subject"], "year": r["year"],
                 "questions": r["n"], "total_marks": int(r["marks"] or 0)}
                for r in papers
            ],
            "by_type": {r["q_type"]: r["n"] for r in types},
        }

    def subjects(self) -> list[str]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT DISTINCT subject FROM questions ORDER BY subject"
            ).fetchall()
        return [r["subject"] for r in rows]

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    # --------------------------------------------------------------- helper

    @staticmethod
    def _row_to_question(row: sqlite3.Row) -> GateQuestion:
        return GateQuestion(
            subject=row["subject"], year=row["year"], q_no=row["q_no"],
            marks=row["marks"], q_type=row["q_type"], stem=row["stem"],
            section=row["section"],
            options=json.loads(row["options_json"]),
            official_answer=row["official_answer"],
            topics=json.loads(row["topics_json"]),
        )
