"""
Hybrid retrieval pipeline:

  1. Vector search (semantic)      -> top TOP_K_VECTOR candidates
  2. BM25 search (lexical/keyword) -> top TOP_K_BM25 candidates
  3. Reciprocal Rank Fusion        -> merge both ranked lists into one
  4. Rerank                        -> rescore the merged candidates with a
                                      finer-grained relevance signal, blended
                                      with the fused score
  5. Abstain                       -> drop everything whose *absolute*
                                      relevance is below CONFIDENCE_THRESHOLD,
                                      so the generator says "I don't know"
                                      instead of improvising

Why hybrid: pure embedding search misses exact matches on rare technical
tokens -- "IDDFS", "B+ tree", "Bessel" -- because they are underrepresented
in the embedding space. Pure BM25 misses paraphrases ("how do I find the
best split" vs "information gain"). Fusing both ranked lists covers both
failure modes, which matters here because a student may search either by
exact syllabus term or by vague description.

Why rerank: steps 1-3 optimize for recall over the whole corpus. Reranking
rescores just the ~16 surviving candidates, which raises precision at the
top of the list where it actually affects the prompt.
"""
from __future__ import annotations

import math
import pickle
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from rank_bm25 import BM25Okapi

from app.config import (
    BM25_INDEX_PATH,
    CONFIDENCE_THRESHOLD,
    RERANK_WEIGHT,
    TOP_K_BM25,
    TOP_K_FINAL,
    TOP_K_VECTOR,
)
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
    score: float
    metadata: dict = field(default_factory=dict)

    @property
    def kind(self) -> str:
        """"question" or "concept" -- lets the prompt builder present a past
        exam question differently from a definition."""
        return self.metadata.get("kind", "document")


class BM25Index:
    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.corpus: list[dict] = []

    def fit(self, chunks: list[Chunk]):
        self.corpus = [
            {"id": c.id, "text": c.text, "source": c.source, "metadata": c.metadata}
            for c in chunks
        ]
        self._rebuild()

    def query(self, query: str, top_k: int, where: dict | None = None) -> list[dict]:
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(_tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits = []
        for i in order:
            if where and not _matches(self.corpus[i]["metadata"], where):
                continue
            hits.append({**self.corpus[i], "score": float(scores[i])})
            if len(hits) >= top_k:
                break
        return hits

    def remove_source(self, source: str):
        self.corpus = [c for c in self.corpus if c["source"] != source]
        self._rebuild()

    def add_chunks(self, chunks: list[Chunk]):
        self.corpus.extend([
            {"id": c.id, "text": c.text, "source": c.source, "metadata": c.metadata}
            for c in chunks
        ])
        self._rebuild()

    def _rebuild(self):
        """BM25 has no native incremental update, so we rebuild from the
        (small, in-memory) corpus list. Fine at question-bank scale; a
        production system would use an incremental lexical index such as
        Elasticsearch/OpenSearch instead."""
        tokenized = [_tokenize(c["text"]) for c in self.corpus]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None
        self._doc_freq = Counter()
        for tokens in tokenized:
            self._doc_freq.update(set(tokens))
        self._n_docs = len(tokenized)

    def doc_freq(self, term: str) -> int:
        return getattr(self, "_doc_freq", Counter()).get(term, 0)

    def idf(self, term: str) -> float:
        """Smoothed IDF over the indexed corpus, used by the reranker so rare
        syllabus terms outweigh common filler words."""
        n = getattr(self, "_n_docs", 0)
        if not n:
            return 1.0
        return math.log((n + 1) / (self.doc_freq(term) + 1)) + 1.0

    def vocabulary_coverage(self, query: str) -> float:
        """Fraction of the query's IDF mass that the corpus actually contains.

        This is the honesty check on semantic search. A TF-IDF/LSA embedding
        can only represent tokens it was fitted on, so a query made mostly of
        unknown words ("best places to visit in Kerala") still embeds to a
        non-zero vector off the back of its one known word and lands a
        confident-looking cosine against an arbitrary document. Scaling
        relevance by this ratio makes the system's confidence track how much
        of the question it can actually see.
        """
        terms = set(_tokenize(query))
        if not terms:
            return 0.0
        total = sum(self.idf(t) for t in terms)
        known = sum(self.idf(t) for t in terms if self.doc_freq(t) > 0)
        return known / total if total else 0.0

    def save(self, path=BM25_INDEX_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"corpus": self.corpus}, f)

    def load(self, path=BM25_INDEX_PATH):
        """Only the corpus is persisted; the BM25 model is rebuilt on load.
        Pickling the fitted BM25Okapi object would tie the on-disk format to
        the library version, which breaks silently on upgrade."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.corpus = state["corpus"]
        self._rebuild()


def _matches(metadata: dict, where: dict) -> bool:
    return all(metadata.get(k) == v for k, v in where.items())


def _reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    """Standard RRF: score(doc) = sum over lists of 1 / (k + rank).

    RRF is used rather than score averaging because BM25 scores and cosine
    distances are on incomparable scales -- normalizing them against each
    other requires per-query calibration, whereas ranks are already
    comparable."""
    fused: dict[str, float] = {}
    for ranked_ids in ranked_lists:
        for rank, doc_id in enumerate(ranked_ids):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return fused


class HybridRetriever:
    def __init__(self, vector_store: VectorStore, embedder: Embedder, bm25_index: BM25Index):
        self.vector_store = vector_store
        self.embedder = embedder
        self.bm25_index = bm25_index

    def _rerank_scores(self, query: str, candidates: dict[str, dict]) -> dict[str, float]:
        """IDF-weighted query-term coverage. Cheap enough to run on every
        request. Swapping in a cross-encoder (e.g. ms-marco-MiniLM) is a
        replacement of this one method -- the caller only needs a mapping
        from doc id to a score in [0, 1]."""
        query_terms = set(_tokenize(query))
        if not query_terms:
            return {doc_id: 0.0 for doc_id in candidates}
        # Denominator counts only terms the corpus actually contains. A term
        # that appears in no document can never be matched by any candidate,
        # so leaving it in the denominator would penalize every document
        # equally and compress the scores that distinguish them. Queries with
        # a lot of unknown vocabulary are handled separately, by scaling the
        # final relevance with vocabulary_coverage().
        attainable = sum(self.bm25_index.idf(t) for t in query_terms
                         if self.bm25_index.doc_freq(t) > 0)
        if not attainable:
            return {doc_id: 0.0 for doc_id in candidates}
        scores = {}
        for doc_id, cand in candidates.items():
            matched = query_terms & set(_tokenize(cand["text"]))
            weight = sum(self.bm25_index.idf(t) for t in matched)
            scores[doc_id] = min(1.0, weight / attainable)
        return scores

    def retrieve(self, query: str, top_k: int = TOP_K_FINAL,
                 where: dict | None = None,
                 min_relevance: float = CONFIDENCE_THRESHOLD) -> list[RetrievedChunk]:
        query_embedding = self.embedder.embed([query])[0]

        # A TF-IDF query whose every token is out of vocabulary transforms to
        # the zero vector. Chroma will still happily return its `top_k`
        # nearest neighbours, all at an identical, meaningless distance --
        # so an off-topic question ("what is the capital of France") comes
        # back looking like a mid-confidence match. Skipping vector search
        # for a degenerate query is what stops that becoming a confident
        # hallucination downstream.
        degenerate_query = not float(np.linalg.norm(query_embedding))
        vector_hits = (
            [] if degenerate_query
            else self.vector_store.query(query_embedding, top_k=TOP_K_VECTOR, where=where)
        )
        bm25_hits = self.bm25_index.query(query, top_k=TOP_K_BM25, where=where)

        candidates: dict[str, dict] = {}
        cosine: dict[str, float] = {}
        for hit in vector_hits:
            meta = dict(hit["metadata"] or {})
            candidates[hit["id"]] = {
                "text": hit["text"],
                "source": meta.get("source", ""),
                "metadata": meta,
            }
            # Embeddings are L2-normalized, so Chroma's squared-L2 distance
            # converts exactly: cos = 1 - d/2.
            cosine[hit["id"]] = max(0.0, min(1.0, 1.0 - hit["distance"] / 2.0))
        for hit in bm25_hits:
            candidates.setdefault(hit["id"], {
                "text": hit["text"],
                "source": hit["source"],
                "metadata": dict(hit["metadata"] or {}),
            })

        if not candidates:
            return []

        fused = _reciprocal_rank_fusion(
            [[h["id"] for h in vector_hits], [h["id"] for h in bm25_hits]]
        )
        rerank = self._rerank_scores(query, candidates)

        # Normalize RRF onto [0, 1] before blending. The theoretical maximum
        # for a document ranked first in both lists is 2/(k+1), so dividing by
        # that puts both signals on the same scale and makes RERANK_WEIGHT
        # mean what its name says.
        rrf_max = 2.0 / 61.0
        final = {
            doc_id: RERANK_WEIGHT * rerank.get(doc_id, 0.0)
            + (1.0 - RERANK_WEIGHT) * min(1.0, fused.get(doc_id, 0.0) / rrf_max)
            for doc_id in candidates
        }

        # Ranking and abstention need different signals, and conflating them
        # was a real bug: RRF is purely rank-based, so the top hit of a
        # *nonsense* query still scores near the maximum. `final` decides the
        # order; `relevance` -- an absolute measure, either semantic
        # similarity or IDF-weighted term coverage -- decides whether any of
        # it is worth showing at all.
        vocab_coverage = self.bm25_index.vocabulary_coverage(query)
        relevance = {
            doc_id: vocab_coverage * max(cosine.get(doc_id, 0.0), rerank.get(doc_id, 0.0))
            for doc_id in candidates
        }

        ranked = sorted(final.items(), key=lambda kv: kv[1], reverse=True)
        return [
            RetrievedChunk(
                id=doc_id,
                text=candidates[doc_id]["text"],
                source=candidates[doc_id]["source"],
                score=round(score, 4),
                metadata=candidates[doc_id]["metadata"],
            )
            for doc_id, score in ranked[:top_k]
            if relevance[doc_id] >= min_relevance
        ]
