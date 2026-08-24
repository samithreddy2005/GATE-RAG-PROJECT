"""
Loads raw documents (.txt, .md, .pdf) from a folder and splits them into
overlapping chunks ready for embedding.

Chunking strategy: recursive-ish split on paragraph boundaries first, and
only falls back to a hard token-count split when a single paragraph is
still too long. This keeps chunks semantically coherent instead of cutting
sentences in half, which is one of the biggest levers on retrieval quality.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

from app.config import CHUNK_OVERLAP, CHUNK_SIZE


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


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
               overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    paragraphs = _split_into_paragraphs(text)
    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_words = 0

    def flush():
        nonlocal buffer, buffer_words
        if not buffer:
            return
        chunk_str = "\n\n".join(buffer).strip()
        if chunk_str:
            chunks.append(Chunk(
                id=str(uuid.uuid4()),
                text=chunk_str,
                source=source,
                chunk_index=len(chunks),
            ))
        buffer, buffer_words = [], 0

    for para in paragraphs:
        para_words = _word_count(para)

        # A single paragraph longer than the chunk size gets hard-split.
        if para_words > chunk_size:
            flush()
            words = para.split()
            for i in range(0, len(words), chunk_size - overlap):
                piece = " ".join(words[i:i + chunk_size])
                chunks.append(Chunk(
                    id=str(uuid.uuid4()),
                    text=piece,
                    source=source,
                    chunk_index=len(chunks),
                ))
            continue

        if buffer_words + para_words > chunk_size:
            flush()
            # carry overlap words forward from the end of the previous chunk
            if chunks:
                prev_words = chunks[-1].text.split()
                overlap_words = prev_words[-overlap:] if overlap else []
                if overlap_words:
                    buffer.append(" ".join(overlap_words))
                    buffer_words += len(overlap_words)

        buffer.append(para)
        buffer_words += para_words

    flush()
    return chunks


def ingest_directory(directory: Path) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() not in (".txt", ".md", ".pdf"):
            continue
        text = load_document(path)
        all_chunks.extend(chunk_text(text, source=path.name))
    return all_chunks
