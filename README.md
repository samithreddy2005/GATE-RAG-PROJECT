# GATE-RAG-PROJECT

A retrieval-augmented study assistant for **GATE** exam preparation, built
around official past papers and their answer keys.

Students preparing for GATE want past questions to practise on and
explanations they can actually follow. Plenty of tools will generate an
explanation. The problem is that a fluent, confident, *wrong* derivation is
worse than no answer at all — a student who memorizes a bad method carries it
into the exam.

So this project treats the **official answer key as ground truth**. Every
generated explanation is parsed for the answer it arrived at and compared
against the key by a deterministic check that no language model takes part in.
If they disagree, the system retries once with the key as a constraint. If
they still disagree, the student sees the derivation with an explicit
**"not verified"** warning rather than a false badge of correctness.

Full design notes, measured results and limitations:
**[rag-project/README.md](rag-project/README.md)**

## Quickstart

Requires **Python 3.12**.

```bash
cd rag-project
python -m venv venv
venv\Scripts\activate            # Windows;  source venv/bin/activate elsewhere
pip install -r requirements.txt
cp .env.example .env             # then paste a key from console.groq.com/keys
python -m uvicorn app.main:app --port 8000 --reload
```

Open **http://127.0.0.1:8000/**.

Practice, browsing, grading and search all work **without** an API key — only
the generated explanations need one.

## What it does

- **Practice mode** — random question sets filtered by subject, topic or type.
  The official answer is never sent to the browser until you submit, and
  grading uses GATE's own marking rules (negative marking on MCQ only).
- **Verified explanations** — step-by-step derivations grounded in a concept
  knowledge base, streamed live, each one checked against the official key
  and badged verified / corrected-on-retry / **not verified**.
- **Ask a doubt** — free-form Q&A grounded in the indexed papers and notes,
  which returns *nothing* rather than improvising when the corpus does not
  cover the question.
- **Analytics** — which topics carry the most marks across past papers, which
  is the question a student actually has when deciding what to revise.

## Measured, not claimed

| | |
|---|---|
| Retrieval recall@1 (20 hand-authored student queries) | 87.5% |
| Off-topic queries correctly refused | 100% |
| Explanations verified against the official key | ~83% (see caveat) |
| Tests | 104 passing |

Reproduce with `python -m pytest` and `python -m eval.run_eval --all`.
Retrieval numbers are deterministic; the generation number is not — the same
seed gave 75.0% and 83.4% on two runs of 12 questions, so read it as "roughly
four in five". Method, caveats and what these numbers do *not* cover are in
the [project README](rag-project/README.md).

## Layout

| Path | |
|---|---|
| [`app/gate_parser.py`](rag-project/app/gate_parser.py) | Subject-agnostic paper parser (MCQ/MSQ/NAT, topic tagging) |
| [`app/question_bank.py`](rag-project/app/question_bank.py) | SQLite store, metadata filters, marks-weighted analytics |
| [`app/build_index.py`](rag-project/app/build_index.py) | Corpus builder + answer-key integrity validation |
| [`app/retriever.py`](rag-project/app/retriever.py) | Hybrid BM25 + vector retrieval with RRF and abstention |
| [`app/gate_generator.py`](rag-project/app/gate_generator.py) | Explanation generation with answer-key verification |
| [`app/main.py`](rag-project/app/main.py) | FastAPI server and streaming endpoints |
| [`data/concepts/`](rag-project/data/concepts/) | Concept notes that ground the *why* |
| [`eval/run_eval.py`](rag-project/eval/run_eval.py) | Retrieval and generation benchmarks |
| [`tests/`](rag-project/tests/) | 104 tests |

## Notes

- Generation runs on [Groq](https://groq.com) (`openai/gpt-oss-120b` by
  default); change it with `GENERATION_MODEL` in `.env`.
- `.env` is gitignored — never commit your API key.
- The included papers are a **partial** extraction, not complete papers;
  `python -m app.build_index` prints the exact coverage gap on every run.
- GATE question papers are the property of their organizing institutes
  (IISc / the IITs) and are used here for educational purposes.
