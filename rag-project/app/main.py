from __future__ import annotations

import shutil
import json
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import DATA_DIR, FRONTEND_DIR, TOP_K_FINAL
from app.embeddings import get_embedder
from app.generator import stream_answer
from app.ingest import ingest_directory
from app.retriever import BM25Index, HybridRetriever
from app.vectorstore import VectorStore

app = FastAPI(title="Real-Time RAG Assistant")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_state = {"embedder": None, "vector_store": None, "bm25_index": None, "retriever": None}


def build_index(directory: Path = DATA_DIR):
    chunks = ingest_directory(directory)
    if not chunks:
        raise ValueError(f"No .txt/.md/.pdf files found in {directory}")

    embedder = get_embedder()
    embedder.fit([c.text for c in chunks])
    embeddings = embedder.embed([c.text for c in chunks])

    vector_store = VectorStore()
    vector_store.reset()
    vector_store.add(chunks, embeddings)

    bm25_index = BM25Index()
    bm25_index.fit(chunks)

    _state["embedder"] = embedder
    _state["vector_store"] = vector_store
    _state["bm25_index"] = bm25_index
    _state["retriever"] = HybridRetriever(vector_store, embedder, bm25_index)
    return len(chunks)


@app.on_event("startup")
def startup():
    # Indexes everything under data/ (raw_papers/ plus anything previously
    # uploaded), so the app is queryable immediately after a restart.
    try:
        n = build_index(DATA_DIR)
        print(f"Startup indexing complete: {n} chunks from {DATA_DIR}")
    except Exception as e:
        print(f"Startup indexing skipped: {e}")


@app.post("/ingest")
async def ingest(files: list[UploadFile] = File(...)):
    upload_dir = DATA_DIR / "uploaded"
    upload_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        dest = upload_dir / f.filename
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
    # Rebuild from DATA_DIR (not just upload_dir) so uploads are added to the
    # existing corpus instead of replacing it.
    n_chunks = build_index(DATA_DIR)
    return {"status": "indexed", "files": [f.filename for f in files], "chunks": n_chunks}


class ChatRequest(BaseModel):
    question: str
    top_k: int = TOP_K_FINAL


@app.post("/chat")
async def chat(req: ChatRequest):
    retriever: HybridRetriever | None = _state["retriever"]
    if retriever is None:
        def err():
            yield f"data: {json.dumps({'type': 'error', 'text': 'No documents indexed yet. Call /ingest first.'})}\n\n"
        return StreamingResponse(err(), media_type="text/event-stream")

    chunks = retriever.retrieve(req.question, top_k=req.top_k)

    def event_stream():
        sources = [
            {"index": i + 1, "source": c.source, "preview": c.text[:180]}
            for i, c in enumerate(chunks)
        ]
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
        for token in stream_answer(req.question, chunks):
            yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/status")
def status():
    vs = _state["vector_store"]
    return {"indexed_chunks": vs.count() if vs else 0}


# Serves the UI from this same server at http://127.0.0.1:8000/ .
# Mounted last so the API routes above take precedence over the "/" catch-all.
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
