from __future__ import annotations

import numpy as np
import chromadb

from app.config import CHROMA_DIR


class VectorStore:
    """Thin wrapper over a persistent Chroma collection.

    Vectors are always supplied by us (embedding_function=None) so Chroma
    never tries to download a default model -- the embedder is chosen in
    app.embeddings and stays the single source of truth.
    """

    def __init__(self, collection_name: str = "gate_corpus"):
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=None
        )

    def reset(self):
        """Drop and recreate the collection. delete_collection raises when
        the collection is absent (a fresh checkout, or a wiped storage/
        directory), so the failure is swallowed -- an absent collection is
        already the desired post-condition."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, embedding_function=None
        )

    def add(self, ids: list[str], texts: list[str], metadatas: list[dict],
            embeddings: np.ndarray, batch_size: int = 2000):
        """Chroma rejects oversized single writes, so adds are batched. The
        batch size is well under the server limit and irrelevant at current
        corpus size, but keeps ingestion of many papers from failing later."""
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            self.collection.upsert(
                ids=ids[start:end],
                embeddings=embeddings[start:end].tolist(),
                documents=texts[start:end],
                metadatas=metadatas[start:end],
            )

    def delete_by_source(self, source: str):
        self.collection.delete(where={"source": source})

    def query(self, query_embedding: np.ndarray, top_k: int,
              where: dict | None = None) -> list[dict]:
        """`where` is a Chroma metadata filter, e.g. {"subject": "DA"}. It is
        applied *before* the nearest-neighbour search, which is what makes
        "Machine Learning questions only" both correct and fast -- filtering
        after the search would silently return fewer than top_k results."""
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where or None,
        )
        if not results["ids"] or not results["ids"][0]:
            return []
        out = []
        for i in range(len(results["ids"][0])):
            out.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return out

    def count(self) -> int:
        return self.collection.count()
