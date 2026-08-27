"""
Loads free-form reference documents (.txt, .md, .pdf) and splits them into
overlapping chunks ready for embedding. This is the *concept* half of the
corpus; the structured past-paper half comes from app.gate_parser and is
indexed as one chunk per question by app.build_index.

Chunking strategy: split on paragraph boundaries first, falling back to a
hard word-count split only when a single paragraph is still too long. This
keeps chunks semantically coherent instead of cutting derivations in half,
which is one of the biggest levers on retrieval quality.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

from app.config import CHUNK_OVERLAP, CHUNK_SIZE

SUPPORTED_SUFFIXES = (".txt", ".md", ".pdf")


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def _chunk_id(source: str, index: int, text: str) -> str:
    """Content-addressed id. Re-indexing unchanged files therefore produces
    the same ids, so an upsert replaces a chunk instead of duplicating it --
    which a random uuid4 per run could never do."""
    digest = hashlib.sha1(f"{source}:{index}:{text}".encode("utf-8")).hexdigest()[:16]
    return f"{source}#{index}#{digest}"


def _read_txt_or_md(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def load_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        return _read_txt_or_md(path)
    if suffix == ".pdf":
        return _read_pdf(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def _split_into_paragraphs(text: str) -> list[str]:
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def _word_count(text: str) -> int:
    return len(text.split())


def chunk_text(text: str, source: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP,
               metadata: dict | None = None) -> list[Chunk]:
    base_metadata = dict(metadata or {})
    base_metadata.setdefault("source", source)
    base_metadata.setdefault("kind", "concept")

    paragraphs = _split_into_paragraphs(text)
    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_words = 0

    def emit(body: str):
        body = body.strip()
        if not body:
            return
        index = len(chunks)
        chunks.append(Chunk(
            id=_chunk_id(source, index, body),
            text=body,
            source=source,
            chunk_index=index,
            metadata=dict(base_metadata),
        ))

    def flush():
        nonlocal buffer, buffer_words
        if buffer:
            emit("\n\n".join(buffer))
        buffer, buffer_words = [], 0

    for para in paragraphs:
        para_words = _word_count(para)

        # A single paragraph longer than the chunk size gets hard-split.
        if para_words > chunk_size:
            flush()
            words = para.split()
            step = max(1, chunk_size - overlap)
            for i in range(0, len(words), step):
                emit(" ".join(words[i:i + chunk_size]))
            continue

        if buffer_words + para_words > chunk_size:
            flush()
            # Carry overlap words forward from the end of the previous chunk
            # so a fact spanning a boundary is retrievable from either side.
            if chunks and overlap:
                overlap_words = chunks[-1].text.split()[-overlap:]
                if overlap_words:
                    buffer.append(" ".join(overlap_words))
                    buffer_words += len(overlap_words)

        buffer.append(para)
        buffer_words += para_words

    flush()
    return chunks


def ingest_directory(directory: Path, metadata: dict | None = None) -> list[Chunk]:
    """Chunk every supported file under `directory`, recursively."""
    all_chunks: list[Chunk] = []
    if not directory.is_dir():
        return all_chunks
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        text = load_document(path)
        if not text.strip():
            continue
        meta = dict(metadata or {})
        meta.setdefault("source", path.name)
        meta.setdefault("path", str(path.relative_to(directory)))
        all_chunks.extend(chunk_text(text, source=path.name, metadata=meta))
    return all_chunks
