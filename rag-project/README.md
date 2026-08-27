# GATE prep assistant — retrieval with answer-key-verified explanations

A retrieval-augmented system for **GATE** exam preparation, built around
official past papers and their answer keys.

The distinguishing idea is not the retrieval. It is that **the answer key is
ground truth, and every generated explanation is checked against it before a
student sees it.** For exam preparation a confidently wrong derivation is
worse than no answer at all — a student who memorizes a plausible but
incorrect method carries that error into the exam. So an explanation is only
labelled correct if a deterministic comparison against the official key says
it is, and no language model participates in that judgement.

---

## Architecture

```
data/raw_papers/<SUBJECT>/<YEAR>/
  ├── *_raw.txt          extracted paper text
  └── *_answer_key.csv   official key (type, key, marks, section)
              │
              ▼
      gate_parser.py ──────► GateQuestion records
              │              (subject, year, q_no, marks, type,
              │               stem, options, official_answer, topics)
              │
              ├──────────────► question_bank.py  (SQLite)
              │                 • metadata filters: subject / year / topic / type
              │                 • topic-frequency and marks-weight analytics
              │                 • validate_paper(): answer-key integrity checks
              │
              ▼
        build_index.py ─── one chunk per question ─┐
                                                   ├─► vector store + BM25
  data/concepts/*.md ─── paragraph chunks ─────────┘        (retriever.py)
                                                              │
                                                              ▼
                                                    hybrid retrieval
                                             BM25 + vector → RRF → rerank
                                                    → abstain if irrelevant
                                                              │
                                                              ▼
                                                   gate_generator.py
                                          generate derivation with concepts
                                          → parse "FINAL ANSWER:"
                                          → compare to official key
                                          → retry once if it disagrees
                                          → flag "unverified" if still wrong
                                                              │
                                                              ▼
                                            FastAPI (main.py) + browser UI
```

Two kinds of content share one index, tagged `kind: question | concept`, so a
single query can surface both *"this was asked in GATE DA 2024"* and *"here is
the theorem that answers it"*.

---

## Measured results

Reproduce everything below with `python -m eval.run_eval --all`.

### Retrieval — `eval/student_queries.json`, 20 hand-authored queries

Queries are phrased the way a student actually searches, and deliberately
share minimal wording with the question stems. A benchmark that feeds a
question's own text back to the retriever measures string matching, not
retrieval, and reports a meaningless 100%.

| Metric | Result |
|---|---|
| recall@1 | 87.5% |
| recall@5 | 87.5% |
| MRR | 0.875 |
| concept precision | 88.2% |
| **abstention rate** | **100%** |

Abstention rate is the number that matters most. It measures off-topic
queries ("what is the capital of France") correctly answered with *nothing*.
A retriever that always returns its top-k is not high-recall — it is
incapable of saying no, and each of those results becomes grounding for a
confident, wrong answer.

### Generation — 12 questions, `--seed 7`, `openai/gpt-oss-120b`

| Metric | Result |
|---|---|
| verified@1 — correct on the first derivation | 66.7% |
| verified@2 — correct only after the correction retry | 16.7% |
| **total verified against the official key** | **83.4%** |
| unverified — shown with an explicit warning | 16.7% |
| truncated / errors | 0 |
| median latency | 16.7 s |
| by type | MCQ 7/9 (78%), NAT 3/3 (100%) |

**Read this number with its error bars.** Sampling is seeded and
reproducible, but generation is not: the same seed on the same corpus
produced **75.0%** and **83.4%** on two runs. At n = 12 a single question
moves the headline by 8 points, so treat this as "roughly four in five" and
raise `-n` before drawing conclusions from a change. Both runs agreed on the
one genuinely hard failure, DA 2024 Q.46 (deriving which functional
dependencies follow from a given set).

The two runs differ mostly because the corpus itself improved between them —
CS 2023 Q.60 started passing once its answer key was corrected (below).

---

## Three things this project got wrong, and what fixed them

These are documented because they are the interesting part of the
engineering, and each one is now covered by a test.

**1. Rank-based fusion cannot express "I don't know."**
Reciprocal Rank Fusion scores by *position*, so the top hit of a nonsense
query scores nearly as high as the top hit of a good one. Thresholding on the
fused score therefore never abstains. Ranking and abstention now use separate
signals: RRF orders the results, while an *absolute* relevance measure
decides whether any of them are worth showing.

**2. An out-of-vocabulary query embeds to the zero vector — and the vector
store answers anyway.**
With TF-IDF/SVD embeddings, a query of entirely unknown words produces a
zero vector. Chroma still returns its top-k neighbours, all at an identical,
meaningless distance, which reads downstream as a mid-confidence match.
Relevance is now scaled by `vocabulary_coverage` — the share of the query's
IDF mass the corpus actually contains — so the system's confidence tracks how
much of the question it can genuinely see. This is what took abstention from
80% to 100%.

**3. The evaluation harness found a bad row in our own answer key.**
GATE CS 2023 Q.60 asks how many rows an SQL query returns. The answer is 2.
The key in `data/` said `2.374 to 2.376` — a value that cannot be a row
count. The model derived 2, was marked unverified, and the failure pointed
straight at the data rather than the model. A malformed key is the most
dangerous bug class here because it never crashes; it just marks correct
derivations wrong forever. `validate_paper()` now runs on every build and
checks that each key matches the shape its question type requires.

---

## Setup

Requires **Python 3.12** (3.14 has no wheels for several dependencies).

```bash
cd rag-project
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

Add an API key — get a free one at [console.groq.com/keys](https://console.groq.com/keys):

```bash
cp .env.example .env             # then edit .env and paste your key
```

Run it:

```bash
python -m uvicorn app.main:app --port 8000 --reload
```

Open **http://127.0.0.1:8000/**. The index builds automatically at startup.

Retrieval, browsing, practice and grading all work **without** an API key —
only the generated explanations need one.

### Commands

```bash
python -m app.build_index          # rebuild the index, print integrity warnings
python -m pytest                   # 104 tests
python -m eval.run_eval --retrieval    # offline, no API key needed
python -m eval.run_eval --generation -n 12 --seed 7
python -m eval.run_eval --all --json eval/results.json
```

---

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/status` | index size, papers, model, key configured |
| `GET` | `/api/subjects` | subjects in the bank |
| `GET` | `/api/topics` | topic frequency and **marks share** |
| `GET` | `/api/questions` | filter by subject / year / topic / type |
| `GET` | `/api/questions/{qid}` | one question with its official answer |
| `GET` | `/api/practice` | random set, **answers withheld** |
| `POST` | `/api/attempt` | grade a submitted answer, GATE marking rules |
| `POST` | `/api/explain` | verified explanation (JSON) |
| `POST` | `/api/explain/stream` | verified explanation (SSE, streams the retry) |
| `POST` | `/api/chat` | free-form grounded Q&A (SSE) |
| `POST` | `/api/ingest` | upload concept notes (.txt/.md/.pdf) |
| `POST` | `/api/rebuild` | rebuild the index |

Interactive docs at `/docs`.

Two design details worth noting. `/api/practice` never sends the official
answer to the browser — otherwise practice mode is trivially cheatable from
devtools. And `/api/attempt` grades with the **same** deterministic comparison
used to verify explanations, so a student's score never depends on a model's
opinion; GATE's marking rules apply, with negative marking on MCQ only.

---

## Adding a paper

Directory layout is the contract — no code change required:

```
data/raw_papers/DA/2025/
  DA_2025_raw.txt
  DA_2025_answer_key.csv     # q_no,type,subject,key,marks
```

Then `python -m app.build_index`, and read the integrity warnings it prints.

A paper without an answer key is skipped rather than half-ingested: without
ground truth it cannot be verified, and an unverifiable explanation is the one
thing this project exists to avoid.

For a **new subject**, add a topic taxonomy to `SUBJECT_TAXONOMIES` in
`app/gate_parser.py`. The parser itself is subject-agnostic — every GATE paper
shares the same `Q.<n>` numbering, marks-banner and MCQ/MSQ/NAT conventions,
confirmed against both the CS and DA papers here.

Keywords need care. Two false positives found during development are
documented in the taxonomy: `"relation"` matched *"recurrence relation"* and
tagged an algorithms question as DBMS, and `"function"` matched *"let f be a
function"* and tagged a linked-list question as Discrete Mathematics. Matching
is word-boundary based, treats hyphens and spaces as interchangeable
(`"decision tree"` matches `"decision-tree"`), and every keyword must be
unambiguous within an exam paper.

---

## Limitations

Stated plainly, because they bound what the numbers above mean.

- **The corpus is a sample, not complete papers.** DA 2024 has 20 parsed
  questions against a 21-row key; CS 2023 has 13 against a 65-row key.
  `python -m app.build_index` reports the exact coverage gap on every run.
  The pipeline handles full papers — the text extraction simply is not
  complete yet.
- **Figures and tables are lost.** Plain-text extraction drops Bayesian
  network diagrams, geometry figures and some data tables. DA 2024 Q.64 gives
  its conditional probability tables only as a figure, so the text alone does
  not determine the answer — the model reached the right value anyway, which
  means it reasoned from a default assumption rather than from the data, and
  a pass on that question should not be read as a solved one. Detecting
  figure-dependent questions at parse time and excluding them from evaluation
  would make the headline metric more trustworthy.
- **The embedder is local TF-IDF + SVD (LSA).** No API key, no model download,
  fully offline — a deliberate trade. It matches on term co-occurrence rather
  than meaning, so it is weaker on paraphrases than a transformer encoder.
  `get_embedder()` in `app/embeddings.py` is the single swap point.
- **The reranker is IDF-weighted lexical overlap**, not a cross-encoder.
  `HybridRetriever._rerank_scores` is the one method to replace.
- **`CONFIDENCE_THRESHOLD` is calibrated to this corpus** (on-topic queries
  score 0.521–0.946, off-topic 0.000–0.397; the threshold sits at 0.45, inside
  that gap). The gap moves with the vocabulary — re-run the retrieval eval
  after materially changing the corpus.
- **Single-process, in-memory index.** BM25 rebuilds fully on ingest and there
  is no auth or rate limiting. Fine for a local study tool; not a deployment.

## Next steps

1. Complete the text extraction for both papers, then add DA 2025/2026.
   2024 was DA's first year, so there is a real ceiling on that subject's data.
2. Swap in a transformer embedder and re-run `eval/run_eval.py` — the harness
   exists precisely so that change can be measured rather than assumed.
3. Persist student attempts to turn `/api/attempt` into weak-topic detection:
   the analytics currently describe the *exam*, not the student.
4. Detect figure-dependent questions at parse time and exclude them from
   practice and evaluation.

---

## Attribution

GATE question papers are the property of their respective organizing
institutes (IISc / the IITs) and are included here for educational use. The
concept notes in `data/concepts/` are original summaries of standard
mathematical and computer-science results written for this project.
