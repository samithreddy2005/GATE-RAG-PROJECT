# GATE DA (Data Science & AI) exam prep assistant — a RAG system with verified, explainable solutions

A retrieval-augmented system for **GATE Data Science and Artificial
Intelligence (DA)** exam preparation, scoped to this one subject. Unlike a
generic "chat with your PDFs" RAG demo, this project is built around a
dataset with **verifiable ground truth** (official answer keys), which
forces real engineering problems: structured extraction from inconsistent
PDFs, subject-agnostic parsing, and — the core piece — verifying that
generated explanations actually arrive at the correct answer before
showing them to a student.

Note on scope: the underlying parser (`app/gate_parser.py`) is
subject-agnostic — every GATE subject shares the same question numbering,
marks-banner, and MCQ/MSQ/NAT convention (confirmed against both CS and DA
official papers below). What's subject-specific is the **topic
taxonomy**, so this project ships with one built for DA (Probability &
Statistics, Linear Algebra, Calculus & Optimization, Programming, Data
Structures & Algorithms, DBMS & Data Warehousing, Machine Learning,
Artificial Intelligence) as well as the original CS taxonomy from initial
prototyping, kept in to demonstrate the multi-subject pattern.

## What's real vs. scaffolded right now

Built and tested against **actual official GATE papers**, fetched live
from official/mirrored sources:

- **GATE DA 2024** (`data/raw_papers/DA/2024/`) — the real, full 65-question
  paper (IISc Bengaluru, the first organizing institute for DA) plus its
  official answer key, including real MSQ multi-answer formatting
  (semicolon-delimited in the source, normalized to comma-delimited here)
  and real NAT range answers.
- **GATE CS 2023** (`data/raw_papers/CS/2023/`) — kept from initial
  prototyping to prove the parser generalizes across subjects without
  changes.

**Parser** (`app/gate_parser.py`) — turns raw extracted PDF text into
structured `GateQuestion` records (subject, year, q_no, marks, type, stem,
options). Handles real messiness found in the actual PDFs: repeated page
headers/footers interleaved mid-question, marks-banner lines that
themselves look like question markers, and duplicate question-number
artifacts from page breaks.

**Answer-key ingestion** — parses the official answer key format
(MCQ/MSQ/NAT, letter keys, numeric ranges like `"0.12 to 0.13"`), and uses
it as the *authoritative* source for question type, overriding text-based
heuristics. Confirmed handling a real formatting inconsistency: the CS key
delimits MSQ answers with `,` while the DA key uses `;` — both normalize
to the same internal format.

**Topic tagging** (`TOPIC_KEYWORDS_DA` in `app/gate_parser.py`) —
keyword-based classifier verified end to end on real DA questions:
correctly tagged Probability & Statistics (7 of 20 test questions —
matches DA's actual syllabus weighting), Machine Learning, DBMS & Data
Warehousing, Artificial Intelligence, etc. Caught and fixed two real
false-positive bugs during testing: naive substring matching tagged a
vocabulary question as "Probability & Statistics" because the keyword
`"mean"` matched inside `"meaning"`, and `"decision tree"` didn't match
the source's `"decision-tree"` phrasing. Fixed with word-boundary regex
that treats spaces/hyphens as interchangeable — a real lesson in why
naive keyword matching breaks in practice.

**Structured question bank** (`app/question_bank.py`) — SQLite store with
metadata filtering (subject, year range, topic) plus topic-frequency
analytics, tested end to end on both subjects: `bank.filter(subject='DA',
topic='Machine Learning')` correctly returns Q17, Q20, Q62.

**Answer verification logic** (`app/gate_generator.py`) — parses a
model's `FINAL ANSWER:` line and checks it against the official key,
including NAT ranges and MSQ sets (order-independent). Unit-tested against
15 cases derived from both subjects' real answer-key formats — all pass.

Scaffolded and ready to wire up, but needs an `ANTHROPIC_API_KEY` to run
live (not available in this sandbox):
- Explanation generation with the retry-on-mismatch verification loop
- Semantic + keyword hybrid retrieval over the question bank (reuses
  `app/embeddings.py` and `app/retriever.py` from the earlier prototype —
  same interfaces, now pointed at flattened `GateQuestion` text instead of
  generic document chunks)
- A concept/syllabus reference base for grounding *why* an answer is
  correct, not just *what* it is
- FastAPI endpoints (`/practice`, `/chat`, `/analytics`) and a student-facing
  UI

## Architecture

```
PDF (official GATE paper)  --->  gate_parser.py  --->  GateQuestion records
Answer key CSV              --->  attach_answer_keys()  --->  official_answer + type
                                          |
                                          v
                                  question_bank.py (SQLite)
                                   - metadata filter (subject/year/topic)
                                   - topic-frequency analytics
                                          |
                                          v
                         retriever.py (hybrid BM25 + vector, reused)
                                          |
                                          v
                   gate_generator.py: generate explanation -> parse FINAL ANSWER
                                    -> compare to official_answer
                                    -> retry once if mismatch
                                    -> flag "unverified" if still mismatched
                                          |
                                          v
                         Step-by-step explanation shown to student,
                         with a visible verified/unverified badge
```

## Next steps to finish it

1. Run the parser across more DA years (2025 and 2026 papers exist —
   same pipeline, just more PDF fetches). 2024 was DA's first year, so
   there's a real ceiling on how far back this subject's data goes.
2. Wire `HybridRetriever` to the question bank's flattened text and add
   metadata pre-filtering before the vector/BM25 stage.
3. Build a small concept knowledge base (short explanations per DA topic —
   Bayes' theorem, PCA, B+ trees, admissible heuristics, etc.) for the
   "why" grounding.
4. Add the FastAPI practice-mode endpoints and a minimal frontend.
5. Add a real evaluation harness: sample N questions, run generation,
   measure verified-on-first-try vs. verified-after-retry vs. never-verified
   rates — this number is your headline project metric.
6. Optional: expand `TOPIC_KEYWORDS_DA` — it's a first pass built from one
   paper's worth of real questions, so less common syllabus areas (e.g.
   specific deep learning architectures, time series) likely need their
   own keywords once more papers are ingested.

## Setup

```bash
pip install -r requirements.txt --break-system-packages
export ANTHROPIC_API_KEY=your_key_here
```
