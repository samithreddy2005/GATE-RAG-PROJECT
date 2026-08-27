# Concept reference base

These notes are the "why" layer of the system. Retrieval pulls the relevant
concept passages alongside the question itself, so a generated explanation is
grounded in stated definitions and theorems rather than in whatever the model
happens to recall.

Each file maps onto one bucket of `TOPIC_KEYWORDS_DA` in
`app/gate_parser.py`, so a question tagged "Linear Algebra" retrieves the
linear algebra notes with a strong lexical signal as well as a semantic one.

**Adding a topic:** drop in a new `.md` file, add matching keywords to the
taxonomy in `app/gate_parser.py`, then re-run `python -m app.build_index`.
Chunking is paragraph-aware (`app/ingest.py`), so keep paragraphs
self-contained — a chunk that ends mid-derivation retrieves poorly.

**Provenance:** these are original summaries of standard, non-copyrightable
mathematical and computer-science results written for this project. They are
not reproduced from any textbook or coaching material.
