"""
Free-form Q&A over the corpus: the "ask a doubt" mode, as opposed to the
verified per-question explanations in app.gate_generator.

There is no answer key to check against here, so the safety mechanism is
different: the model is constrained to the retrieved passages and instructed
to say when they do not cover the question, and retrieval itself abstains
below CONFIDENCE_THRESHOLD before generation is ever reached.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator

from groq import Groq, GroqError

from app.config import (
    GENERATION_MAX_TOKENS,
    GENERATION_MODEL,
    GENERATION_TEMPERATURE,
    GROQ_API_KEY,
)
from app.retriever import RetrievedChunk

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a GATE exam preparation tutor. Answer using ONLY \
the provided passages, which are a mix of past GATE exam questions and \
concept reference notes.

Rules:
- If the passages do not contain the answer, say so plainly and name what is \
missing. Never invent a formula, a result, or a past-paper question.
- Cite passages inline by number, like [1] or [2].
- When a passage is a past exam question, mention the exam and year so the \
student knows it actually appeared.
- Teach the method, not just the result: a student should be able to redo it.
- Write maths in plain text: x^2, sqrt(n), lambda, <=, 1/2. Do NOT use LaTeX, \
backslash commands, dollar signs, or markdown tables -- the output is displayed \
as plain text and any markup is shown to the student literally.
"""


def _build_context_block(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        label = c.metadata.get("source") or c.source
        kind = "past exam question" if c.kind == "question" else "concept notes"
        parts.append(f"[{i}] ({kind}) {label}\n{c.text}")
    return "\n\n---\n\n".join(parts)


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    return (
        f"Passages:\n\n{_build_context_block(chunks)}\n\n"
        f"---\n\nStudent's question: {question}\n\n"
        f"Answer using only the passages above, with inline citations like [1]."
    )


def stream_answer(question: str, chunks: list[RetrievedChunk]) -> Iterator[str]:
    if not GROQ_API_KEY:
        yield ("No GROQ_API_KEY is configured. Add one to rag-project/.env "
               "and restart the server to enable answers.")
        return

    if not chunks:
        yield ("Nothing in the indexed papers or concept notes is close enough "
               "to that question for me to answer it reliably. Try rephrasing "
               "with the syllabus term you have in mind, or add notes on the "
               "topic under data/concepts/ and rebuild the index.")
        return

    try:
        client = Groq(api_key=GROQ_API_KEY)
        stream = client.chat.completions.create(
            model=GENERATION_MODEL,
            max_tokens=GENERATION_MAX_TOKENS,
            temperature=GENERATION_TEMPERATURE,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(question, chunks)},
            ],
            stream=True,
        )
        for chunk in stream:
            # Defensive: a keepalive or malformed frame must not kill an
            # answer that is already half-streamed to the browser.
            if not chunk.choices:
                continue
            text = chunk.choices[0].delta.content
            if text:
                yield text
    except GroqError as e:
        log.exception("Groq streaming call failed")
        yield f"\n\n[The language model call failed: {e}]"
