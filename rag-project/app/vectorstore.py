from __future__ import annotations

import numpy as np
import chromadb

from app.config import CHROMA_DIR
from app.ingest import Chunk


class VectorStore:
    def __init__(self, collection_name: str = "documents"):
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        # embedding_function=None: we always supply our own precomputed
        # vectors, so Chroma never tries to download a default model.
        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=None
        )

    def reset(self):
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name="documents", embedding_function=None
        )

    def add(self, chunks: list[Chunk], embeddings: np.ndarray):
        self.collection.add(
            ids=[c.id for c in chunks],
            embeddings=embeddings.tolist(),
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, "chunk_index": c.chunk_index} for c in chunks],
        )

    def delete_by_source(self, source: str):
        self.collection.delete(where={"source": source})

    def query(self, query_embedding: np.ndarray, top_k: int) -> list[dict]:
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
        )
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
