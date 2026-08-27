"""Central configuration. Every tunable lives here so experiments are a
one-line change rather than a hunt through the codebase."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_PAPERS_DIR = DATA_DIR / "raw_papers"
CONCEPTS_DIR = DATA_DIR / "concepts"
UPLOAD_DIR = DATA_DIR / "uploaded"
FRONTEND_DIR = BASE_DIR / "frontend"
STORAGE_DIR = BASE_DIR / "storage"

# Reads BASE_DIR/.env so secrets stay out of version control.
load_dotenv(BASE_DIR / ".env")

CHROMA_DIR = STORAGE_DIR / "chroma"
BM25_INDEX_PATH = STORAGE_DIR / "bm25.pkl"
EMBEDDER_PATH = STORAGE_DIR / "embedder.pkl"
QUESTION_DB_PATH = STORAGE_DIR / "question_bank.sqlite3"

CHUNK_SIZE = 350          # words per chunk for free-form concept notes
CHUNK_OVERLAP = 60        # words of overlap between consecutive chunks

TOP_K_VECTOR = 8          # candidates pulled from vector search
TOP_K_BM25 = 8            # candidates pulled from keyword search
TOP_K_FINAL = 4           # passages actually sent to the LLM after reranking

# Weighting between the recall-oriented RRF score and the precision-oriented
# rerank score. 0.0 = pure RRF, 1.0 = pure rerank. See retriever._final_ranking.
RERANK_WEIGHT = 0.6

# Absolute relevance below which retrieval is treated as having found
# nothing useful, so the student is told so rather than being handed an
# improvised answer.
#
# Measured as vocabulary_coverage * max(cosine similarity, IDF-weighted term
# coverage) -- deliberately NOT the fused RRF score, which is rank-based and
# therefore near-maximal for the top hit of any query, nonsense included.
#
# Calibrated against eval/student_queries.json on this corpus: the 20
# on-topic queries score 0.521-0.946, the 5 off-topic ones 0.000-0.397. The
# threshold sits in that gap rather than on either edge, so re-run
# `python -m eval.run_eval --retrieval` after changing the corpus -- the gap
# moves with the vocabulary.
CONFIDENCE_THRESHOLD = 0.45

EMBEDDING_DIM = 256       # dimensionality of the local SVD embedding space

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "openai/gpt-oss-120b")
# Reasoning models (gpt-oss and similar) spend a large share of their budget
# on internal reasoning before emitting any visible text. At 1400 the
# evaluation harness showed finish_reason="length" with an EMPTY visible
# answer on multi-step questions -- the derivation was cut off before it ever
# reached its "FINAL ANSWER:" line, which then looked like a wrong answer
# rather than a truncated one. Keep this generous.
GENERATION_MAX_TOKENS = int(os.environ.get("GENERATION_MAX_TOKENS", "4096"))
GENERATION_TEMPERATURE = float(os.environ.get("GENERATION_TEMPERATURE", "0.2"))

# Answer-key sentinels that are not real answers. GATE publishes "MTA"
# (Marks To All) when a question is dropped and every candidate is awarded
# the marks. Such rows have no correct option, so they can never be
# verified and must be excluded from both verification and evaluation.
UNANSWERABLE_KEYS = {"MTA", "", "NA", "-"}

STORAGE_DIR.mkdir(parents=True, exist_ok=True)
