from __future__ import annotations

from collections.abc import Iterator

from groq import Groq

from app.config import GENERATION_MODEL, GROQ_API_KEY
from app.retriever import RetrievedChunk

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using \
ONLY the provided context passages. Rules:
- If the answer is not contained in the context, say you don't know based \
on the available documents. Never make up information.
- Cite sources inline using the passage numbers given, like [1] or [2].
- Be concise and direct.
"""


def _build_context_block(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[{i}] Source: {c.source}\n{c.text}")
    return "\n\n---\n\n".join(parts)


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context = _build_context_block(chunks)
    return (
        f"Context passages:\n\n{context}\n\n"
        f"---\n\nQuestion: {question}\n\n"
        f"Answer using only the context above, with inline citations like [1]."
    )


def stream_answer(question: str, chunks: list[RetrievedChunk]) -> Iterator[str]:
    if not GROQ_API_KEY:
        yield "[No GROQ_API_KEY set. Add one to your environment to enable generation.]"
        return

    if not chunks:
        yield "I couldn't find anything relevant in the indexed documents to answer that."
        return

    client = Groq(api_key=GROQ_API_KEY)
    prompt = build_prompt(question, chunks)

    stream = client.chat.completions.create(
        model=GENERATION_MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        stream=True,
    )
    for chunk in stream:
        text = chunk.choices[0].delta.content
        if text:
            yield text
