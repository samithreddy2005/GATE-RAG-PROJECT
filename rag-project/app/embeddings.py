"""
Embedding layer, built behind a small interface so the backing model is a
one-line swap.

Default implementation: TF-IDF + Truncated SVD (i.e. classic LSA). It's
fully local, needs no API key and no model download, which matters in
sandboxed / offline environments. It's a legitimate, well-understood
embedding technique -- not a toy -- though a transformer-based encoder
(OpenAI text-embedding-3, Cohere embed-v3, or an open-source model like
bge-large) will retrieve better on paraphrased queries because it captures
meaning rather than term co-occurrence. Swap it in by implementing the same
`Embedder` interface below and pointing `get_embedder()` at it.
"""
from __future__ import annotations

import pickle
from abc import ABC, abstractmethod

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from app.config import EMBEDDER_PATH, EMBEDDING_DIM


class Embedder(ABC):
    @abstractmethod
    def fit(self, texts: list[str]) -> None: ...

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray: ...

    @abstractmethod
    def save(self, path=EMBEDDER_PATH) -> None: ...

    @abstractmethod
    def load(self, path=EMBEDDER_PATH) -> None: ...


class TfidfSvdEmbedder(Embedder):
    """Local LSA-style embedder: TF-IDF vectors compressed with SVD, then
    L2-normalized so cosine similarity == dot product."""

    def __init__(self, n_components: int = EMBEDDING_DIM):
        self.n_components = n_components
        self.vectorizer = TfidfVectorizer(
            max_features=50_000, ngram_range=(1, 2), stop_words="english"
        )
        self.svd: TruncatedSVD | None = None

    def fit(self, texts: list[str]) -> None:
        tfidf = self.vectorizer.fit_transform(texts)
        n_components = min(self.n_components, max(2, tfidf.shape[1] - 1), tfidf.shape[0] - 1)
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.svd.fit(tfidf)

    def embed(self, texts: list[str]) -> np.ndarray:
        if self.svd is None:
            raise RuntimeError("Embedder not fitted. Call fit() or load() first.")
        tfidf = self.vectorizer.transform(texts)
        vectors = self.svd.transform(tfidf)
        return normalize(vectors)

    def save(self, path=EMBEDDER_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "svd": self.svd}, f)

    def load(self, path=EMBEDDER_PATH) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.vectorizer = state["vectorizer"]
        self.svd = state["svd"]


def get_embedder() -> Embedder:
    """Single place to swap embedding backends, e.g. return an
    OpenAIEmbedder() or CohereEmbedder() that implements the same
    interface but calls a hosted API instead."""
    return TfidfSvdEmbedder()
