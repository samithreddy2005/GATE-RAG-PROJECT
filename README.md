# GATE-RAG-PROJECT

A retrieval-augmented assistant for **GATE Data Science & AI (DA)** exam prep,
built around official question papers and answer keys as verifiable ground truth.

Unlike a generic "chat with your PDFs" demo, this project is built on a dataset
with real answer keys — which forces the harder engineering problems: structured
extraction from inconsistent PDFs, subject-agnostic parsing, and verifying that a
generated explanation actually reaches the correct answer before a student sees it.

Full design notes, architecture, and project status: **[rag-project/README.md](rag-project/README.md)**

## Quickstart

Requires **Python 3.12** (3.14 has no wheels for several dependencies).

```bash
cd rag-project
python -m venv venv
venv\Scripts\activate          # Windows;  source venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

Add your API key:

```bash
cp .env.example .env           # then edit .env and paste your key
```

Get a free key at [console.groq.com/keys](https://console.groq.com/keys).

Run it:

```bash
python -m uvicorn app.main:app --port 8000 --reload
```

Open **http://127.0.0.1:8000/** — the API serves the UI, and the corpus in
`data/` is indexed automatically at startup.

## What's here

| Path | |
|---|---|
| `app/main.py` | FastAPI server — `/chat` (streaming, with citations), `/ingest`, `/status` |
| `app/retriever.py` | Hybrid retrieval: BM25 + vector search fused with RRF |
| `app/gate_parser.py` | Subject-agnostic GATE paper parser (MCQ/MSQ/NAT, topic tagging) |
| `app/gate_generator.py` | Explanation generation verified against the official answer key |
| `app/question_bank.py` | SQLite question store with metadata filtering and analytics |
| `data/raw_papers/` | Official GATE DA 2024 and CS 2023 papers + answer keys |

## Notes

- Generation runs on [Groq](https://groq.com) (`openai/gpt-oss-120b` by default).
  Change the model with `GENERATION_MODEL` in `.env`.
- `.env` is gitignored — never commit your API key.
- GATE question papers included here are the property of their respective
  organizing institutes (IISc / IITs) and are used for educational purposes.
