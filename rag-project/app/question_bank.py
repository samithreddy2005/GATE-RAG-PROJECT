"""
Structured question bank: SQLite holds the ground-truth structured
records (subject, year, topic, official answer), while the existing
embeddings/BM25 machinery (app.embeddings, app.retriever) indexes the
flattened question text for semantic + keyword search.

This split matters: metadata filters (year range, topic, subject) run as
a fast SQL WHERE clause, and only the *remaining* candidates go through
similarity search. That's both faster and more precise than trying to
encode "year > 2020" into a vector.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from app.gate_parser import GateQuestion

SCHEMA = """
CREATE TABLE IF NOT EXISTS questions (
    id TEXT PRIMARY KEY,
    subject TEXT NOT NULL,
    year INTEGER NOT NULL,
    q_no INTEGER NOT NULL,
    marks INTEGER NOT NULL,
    q_type TEXT NOT NULL,
    stem TEXT NOT NULL,
    options_json TEXT NOT NULL,
    official_answer TEXT NOT NULL,
    topics_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_subject_year ON questions(subject, year);
"""


class QuestionBank:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.executescript(SCHEMA)

    @staticmethod
    def _id(q: GateQuestion) -> str:
        return f"{q.subject}-{q.year}-{q.q_no}"

    def upsert(self, questions: list[GateQuestion]) -> list[str]:
        ids = []
        for q in questions:
            qid = self._id(q)
            ids.append(qid)
            self.conn.execute(
                """INSERT INTO questions
                   (id, subject, year, q_no, marks, q_type, stem, options_json, official_answer, topics_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                     marks=excluded.marks, q_type=excluded.q_type, stem=excluded.stem,
                     options_json=excluded.options_json, official_answer=excluded.official_answer,
                     topics_json=excluded.topics_json""",
                (qid, q.subject, q.year, q.q_no, q.marks, q.q_type, q.stem,
                 json.dumps(q.options), q.official_answer, json.dumps(q.topics)),
            )
        self.conn.commit()
        return ids

    def get(self, qid: str) -> GateQuestion | None:
        row = self.conn.execute("SELECT * FROM questions WHERE id=?", (qid,)).fetchone()
        return self._row_to_question(row) if row else None

    def filter(self, subject: str | None = None, year_min: int | None = None,
               year_max: int | None = None, topic: str | None = None) -> list[GateQuestion]:
        clauses, params = [], []
        if subject:
            clauses.append("subject = ?"); params.append(subject)
        if year_min:
            clauses.append("year >= ?"); params.append(year_min)
        if year_max:
            clauses.append("year <= ?"); params.append(year_max)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.conn.execute(f"SELECT * FROM questions {where}", params).fetchall()
        results = [self._row_to_question(r) for r in rows]
        if topic:
            results = [q for q in results if topic in q.topics]
        return results

    def topic_frequency(self, subject: str | None = None) -> dict[str, int]:
        """Powers the 'which topics show up most' analytics feature."""
        rows = self.filter(subject=subject)
        counts: dict[str, int] = {}
        for q in rows:
            for t in q.topics:
                counts[t] = counts.get(t, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))

    def _row_to_question(self, row) -> GateQuestion:
        cols = [d[0] for d in self.conn.execute("SELECT * FROM questions LIMIT 0").description]
        d = dict(zip(cols, row))
        return GateQuestion(
            subject=d["subject"], year=d["year"], q_no=d["q_no"], marks=d["marks"],
            q_type=d["q_type"], stem=d["stem"], options=json.loads(d["options_json"]),
            official_answer=d["official_answer"], topics=json.loads(d["topics_json"]),
        )
