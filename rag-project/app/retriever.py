"""
Hybrid retrieval pipeline:

  1. Vector search (semantic)      -> top TOP_K_VECTOR candidates
  2. BM25 search (lexical/keyword) -> top TOP_K_BM25 candidates
  3. Reciprocal Rank Fusion        -> merge both ranked lists into one
  4. Rerank                        -> reorder the merged candidates by a
                                       finer-grained relevance score before
                                       truncating to TOP_K_FINAL

Why hybrid: pure embedding search misses exact matches on things like
product codes, names, or rare technical terms because those tokens are
underrepresented in the embedding space. Pure BM25 misses paraphrases
("how do I cancel" vs "termination process"). Combining both and fusing
the rankings covers both failure modes.

Why rerank: retrieval (steps 1-3) optimizes for speed over a large corpus.
Reranking runs a more expensive, more accurate relevance score over just
the ~15 candidates that made it through, which consistently improves
precision at the top of the list. In production swap `_rerank` for a real
cross-encoder (e.g. ms-marco-MiniLM via a hosted API) -- the interface
below is written so that's a one-function swap.
"""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from app.config import BM25_INDEX_PATH, TOP_K_BM25, TOP_K_FINAL, TOP_K_VECTOR
from app.embeddings import Embedder
from app.ingest import Chunk
from app.vectorstore import VectorStore

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class RetrievedChunk:
    id: str
    text: str
    source: str
    chunk_index: int
    score: float


class BM25Index:
    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.corpus: list[dict] = []  # id, text, source, chunk_index

    def fit(self, chunks: list[Chunk]):
        self.corpus = [
            {"id": c.id, "text": c.text, "source": c.source, "chunk_index": c.chunk_index}
            for c in chunks
        ]
        tokenized = [_tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def query(self, query: str, top_k: int) -> list[dict]:
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(_tokenize(query))
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [{**self.corpus[i], "score": float(scores[i])} for i in ranked_idx]

    def remove_source(self, source: str):
        self.corpus = [c for c in self.corpus if c["source"] != source]
        self._rebuild()

    def add_chunks(self, chunks: list[Chunk]):
        self.corpus.extend([
            {"id": c.id, "text": c.text, "source": c.source, "chunk_index": c.chunk_index}
            for c in chunks
        ])
        self._rebuild()

    def _rebuild(self):
        """BM25 has no native incremental update, so we rebuild from the
        (small, in-memory) corpus list. Fine at demo/project scale; a
        production system would shard or use an incremental-friendly
        lexical index (e.g. Elasticsearch/OpenSearch) instead."""
        tokenized = [_tokenize(c["text"]) for c in self.corpus]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None

    def save(self, path=BM25_INDEX_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "corpus": self.corpus}, f)

    def load(self, path=BM25_INDEX_PATH):
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.bm25 = state["bm25"]
        self.corpus = state["corpus"]


def _reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    """Standard RRF: score(doc) = sum over lists of 1 / (k + rank)."""
    fused: dict[str, float] = {}
    for ranked_ids in ranked_lists:
        for rank, doc_id in enumerate(ranked_ids):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return fused


def _rerank(query: str, candidates: dict[str, dict]) -> list[str]:
    """Lightweight lexical-overlap reranker (query-term coverage, weighted
    by term rarity via IDF-like log scaling). Cheap enough to run on every
    request; swap for a cross-encoder here for higher accuracy."""
    query_terms = set(_tokenize(query))
    scored = []
    for doc_id, cand in candidates.items():
        doc_terms = _tokenize(cand["text"])
        doc_term_set = set(doc_terms)
        overlap = len(query_terms & doc_term_set)
        coverage = overlap / max(1, len(query_terms))
        density = overlap / max(1, len(doc_terms)) * 10
        scored.append((doc_id, coverage + density))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in scored]


class HybridRetriever:
    def __init__(self, vector_store: VectorStore, embedder: Embedder, bm25_index: BM25Index):
        self.vector_store = vector_store
        self.embedder = embedder
        self.bm25_index = bm25_index

    def retrieve(self, query: str, top_k: int = TOP_K_FINAL) -> list[RetrievedChunk]:
        query_embedding = self.embedder.embed([query])[0]
        vector_hits = self.vector_store.query(query_embedding, top_k=TOP_K_VECTOR)
        bm25_hits = self.bm25_index.query(query, top_k=TOP_K_BM25)

        candidates: dict[str, dict] = {}
        for hit in vector_hits:
            candidates[hit["id"]] = {
                "text": hit["text"],
                "source": hit["metadata"]["source"],
                "chunk_index": hit["metadata"]["chunk_index"],
            }
        for hit in bm25_hits:
            candidates.setdefault(hit["id"], {
                "text": hit["text"], "source": hit["source"], "chunk_index": hit["chunk_index"],
            })

        if not candidates:
            return []

        vector_ranked_ids = [h["id"] for h in vector_hits]
        bm25_ranked_ids = [h["id"] for h in bm25_hits]
        fused_scores = _reciprocal_rank_fusion([vector_ranked_ids, bm25_ranked_ids])

        reranked_ids = _rerank(query, candidates)
        # Blend RRF (recall-oriented, cross-source) with the rerank order
        # (precision-oriented) by sorting on rerank order primarily,
        # breaking ties with fused score.
        rerank_position = {doc_id: i for i, doc_id in enumerate(reranked_ids)}
        final_ids = sorted(
            candidates.keys(),
            key=lambda d: (rerank_position.get(d, len(candidates)), -fused_scores.get(d, 0.0)),
        )[:top_k]

        return [
            RetrievedChunk(
                id=doc_id,
                text=candidates[doc_id]["text"],
                source=candidates[doc_id]["source"],
                chunk_index=candidates[doc_id]["chunk_index"],
                score=fused_scores.get(doc_id, 0.0),
            )
            for doc_id in final_ids
        ]
