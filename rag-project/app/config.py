import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FRONTEND_DIR = BASE_DIR / "frontend"

# Reads BASE_DIR/.env so secrets stay out of version control.
load_dotenv(BASE_DIR / ".env")
CHROMA_DIR = BASE_DIR / "storage" / "chroma"
BM25_INDEX_PATH = BASE_DIR / "storage" / "bm25.pkl"
EMBEDDER_PATH = BASE_DIR / "storage" / "embedder.pkl"

CHUNK_SIZE = 350          # tokens (approx, whitespace-split)
CHUNK_OVERLAP = 60        # tokens of overlap between consecutive chunks

TOP_K_VECTOR = 8          # candidates pulled from vector search
TOP_K_BM25 = 8            # candidates pulled from keyword search
TOP_K_FINAL = 4           # chunks actually sent to the LLM after reranking

CONFIDENCE_THRESHOLD = 0.02   # fused RRF score below this triggers reformulation/abstention
WATCH_DIR_NAME = "sample_docs"  # subfolder of DATA_DIR watched for live changes

EMBEDDING_DIM = 256        # dimensionality of the local SVD embedding space

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "openai/gpt-oss-120b")

CHROMA_DIR.parent.mkdir(parents=True, exist_ok=True)
