"""
Generates a step-by-step explanation for a GATE question, then verifies the
explanation's derived answer against the official answer key before it is
shown to the student.

This is the piece that makes the project more than a wrapper around an LLM.
For exam preparation a confidently wrong derivation is worse than no answer
at all -- a student who memorizes a plausible but incorrect method carries
that error into the exam. So nothing is presented as correct unless it was
checked against ground truth.

Flow:
  1. Generate an explanation grounded in the question plus retrieved concept
     context, ending in a machine-parseable "FINAL ANSWER: X" line.
  2. Parse the model's final answer out of that line.
  3. Compare it to the official answer key. No LLM is involved in that
     comparison -- it is a deterministic string/interval check, which is the
     whole point: an LLM grading its own work is not verification.
  4. On disagreement, retry once with the official answer named explicitly.
     A model that reasons its way to the right answer on the second pass has
     usually produced a genuinely better derivation.
  5. If it still disagrees, the explanation is surfaced with an explicit
     "unverified" flag rather than silently trusted. That is the abstention
     behavior for this domain.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, field

from groq import Groq, GroqError

from app.config import (
    GENERATION_MAX_TOKENS,
    GENERATION_MODEL,
    GENERATION_TEMPERATURE,
    GROQ_API_KEY,
)
from app.gate_parser import GateQuestion
from app.retriever import RetrievedChunk

log = logging.getLogger(__name__)

FINAL_ANSWER_RE = re.compile(r"FINAL ANSWER:\s*(.+)", re.IGNORECASE)
# Official NAT keys look like "2.374 to 2.376", sometimes with alternatives
# joined by OR: "2 to 2 OR 4 to 4".
NAT_RANGE_RE = re.compile(r"^\s*(-?[\d.]+)\s+TO\s+(-?[\d.]+)\s*$", re.IGNORECASE)

SYSTEM_PROMPT = """You are a GATE exam tutor. Given a question, its type \
(MCQ/MSQ/NAT), and reference material on the underlying concepts, produce a \
clear, step-by-step derivation a student can reproduce on their own.

Rules:
- Work through the reasoning in numbered steps, naming the concept or theorem \
each step relies on.
- Where the reference material supports a step, cite it inline as [1], [2], \
matching the numbered passages you were given.
- Do not skip algebra a student would need to follow the logic.
- For MCQ/MSQ, say briefly why the wrong options are wrong.
- Write maths in plain text: x^2, sqrt(n), lambda, <=, |M|, 1/2. Do NOT use \
LaTeX, backslash commands, dollar signs, or markdown tables -- the output is \
displayed as plain text and any markup is shown to the student literally.
- End with a line in exactly this format: "FINAL ANSWER: <answer>"
  - MCQ: a single letter, e.g. "FINAL ANSWER: B"
  - MSQ: comma-separated letters in alphabetical order, e.g. "FINAL ANSWER: B,C,D"
  - NAT: the numeric value only, no units, e.g. "FINAL ANSWER: 4096"
"""


@dataclass
class Explanation:
    text: str
    derived_answer: str
    official_answer: str
    verified: bool
    attempts: int = 1
    # verified | corrected | unverified | truncated | no_key | error
    status: str = "verified"
    sources: list[dict] = field(default_factory=list)

    @property
    def student_note(self) -> str:
        """One honest sentence shown next to the badge in the UI."""
        return {
            "verified": "Derivation matched the official answer key on the first attempt.",
            "corrected": "The first derivation disagreed with the official key; this is the corrected second attempt.",
            "unverified": ("This derivation does NOT match the official answer key. "
                           "Treat it as a hint, not a solution, and check the key yourself."),
            "no_key": "This question has no single correct answer in the official key (e.g. marks awarded to all).",
            "truncated": ("The derivation was cut off before it reached an answer "
                          "(token limit). Raise GENERATION_MAX_TOKENS and retry."),
            "error": "Generation failed, so nothing could be verified.",
        }[self.status]


def build_concept_context(chunks: list[RetrievedChunk]) -> tuple[str, list[dict]]:
    """Numbered passage block plus the matching source list for citations."""
    parts, sources = [], []
    for i, c in enumerate(chunks, start=1):
        label = c.metadata.get("source") or c.source
        parts.append(f"[{i}] {label}\n{c.text}")
        sources.append({
            "index": i,
            "source": label,
            "kind": c.kind,
            "score": c.score,
            "preview": c.text[:200],
        })
    return "\n\n---\n\n".join(parts), sources


def build_prompt(q: GateQuestion, concept_context: str = "",
                  correction_note: str = "") -> str:
    parts = [
        f"Exam: GATE {q.subject} {q.year}, Question {q.q_no}",
        f"Question type: {q.q_type}, marks: {q.marks}",
        f"Question: {q.stem}",
    ]
    if q.options:
        options_block = "\n".join(f"({k}) {v}" for k, v in sorted(q.options.items()))
        parts.append(f"Options:\n{options_block}")
    if concept_context:
        parts.append(f"Reference material:\n{concept_context}")
    if correction_note:
        parts.append(correction_note)
    parts.append("Explain step by step, then give the FINAL ANSWER line.")
    return "\n\n".join(parts)


def parse_final_answer(text: str) -> str:
    """Take the *last* FINAL ANSWER line. Models sometimes restate the
    format mid-explanation; the concluding line is the real one."""
    matches = FINAL_ANSWER_RE.findall(text)
    if not matches:
        return ""
    answer = matches[-1].strip().upper()
    # Strip trailing prose and markdown emphasis the model may append.
    answer = re.sub(r"[*_`]", "", answer).strip().rstrip(".")
    return answer.replace(" ", "")


def answers_match(derived: str, official: str) -> bool:
    """Deterministic ground-truth comparison. Handles the three official
    answer-key shapes:

      MCQ  "B"              -> exact letter match
      MSQ  "A,B,C"          -> set equality, so order does not matter
      NAT  "0.12 to 0.13"   -> interval membership, optionally several
                               alternatives joined by OR

    Range detection is anchored on the "<num> TO <num>" shape rather than a
    substring test for "TO": a bare `"TO" in official` check would misfire
    on any key that happened to contain those letters.
    """
    derived = (derived or "").strip().upper()
    official = (official or "").strip().upper()
    if not derived or not official:
        return False

    alternatives = [alt.strip() for alt in re.split(r"\s+OR\s+", official)]
    if any(NAT_RANGE_RE.match(alt) for alt in alternatives):
        try:
            value = float(derived)
        except ValueError:
            return False
        for alt in alternatives:
            m = NAT_RANGE_RE.match(alt)
            if not m:
                continue
            lo, hi = float(m.group(1)), float(m.group(2))
            if min(lo, hi) <= value <= max(lo, hi):
                return True
        return False

    if "," in official or "," in derived:
        return {p for p in official.split(",") if p} == {p for p in derived.split(",") if p}

    # A bare numeric key ("3") should still compare numerically, so that a
    # derived "3.0" is not rejected on formatting alone.
    try:
        return float(derived) == float(official)
    except ValueError:
        return derived == official


def _call_model(prompt: str, stream: bool = False):
    client = Groq(api_key=GROQ_API_KEY)
    return client.chat.completions.create(
        model=GENERATION_MODEL,
        max_tokens=GENERATION_MAX_TOKENS,
        temperature=GENERATION_TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        stream=stream,
    )


def _complete(prompt: str) -> tuple[str, str]:
    """Returns (text, finish_reason). The finish reason is carried out rather
    than discarded because "length" means the derivation was cut off before
    its FINAL ANSWER line -- an infrastructure failure that must not be
    reported to a student as a wrong answer."""
    response = _call_model(prompt)
    choice = response.choices[0]
    return (choice.message.content or ""), (choice.finish_reason or "")


def stream_tokens(prompt: str) -> Iterator[str]:
    """Token iterator with defensive unpacking -- a malformed or empty delta
    must not abort a half-written explanation."""
    for chunk in _call_model(prompt, stream=True):
        if not chunk.choices:
            continue
        text = chunk.choices[0].delta.content
        if text:
            yield text


def correction_note(derived: str, official: str) -> str:
    return (
        f"Your previous derivation concluded FINAL ANSWER: {derived or '(none given)'}, "
        f"but the official GATE answer key gives {official}. Re-derive the solution so "
        f"the reasoning genuinely arrives at the official answer, and point out where "
        f"the earlier attempt went wrong. If you are confident the official key is "
        f"itself mistaken, say so explicitly instead of forcing an incorrect derivation."
    )


def generate_verified_explanation(q: GateQuestion,
                                  chunks: list[RetrievedChunk] | None = None
                                  ) -> Explanation:
    concept_context, sources = build_concept_context(chunks or [])

    if not GROQ_API_KEY:
        return Explanation(
            text="No GROQ_API_KEY is configured, so no explanation could be generated. "
                 "Add one to rag-project/.env and restart the server.",
            derived_answer="", official_answer=q.official_answer,
            verified=False, status="error", sources=sources,
        )

    if not q.is_answerable:
        return Explanation(
            text=f"GATE published no single correct answer for this question "
                 f"(official key: {q.official_answer or 'missing'}). It is excluded "
                 f"from verification, so no explanation is generated.",
            derived_answer="", official_answer=q.official_answer,
            verified=False, status="no_key", sources=sources,
        )

    try:
        text, finish_reason = _complete(build_prompt(q, concept_context))
        if finish_reason == "length" and not parse_final_answer(text):
            return Explanation(
                text=text or "", derived_answer="",
                official_answer=q.official_answer, verified=False,
                status="truncated", sources=sources,
            )
        derived = parse_final_answer(text)
        if answers_match(derived, q.official_answer):
            return Explanation(text=text, derived_answer=derived,
                               official_answer=q.official_answer, verified=True,
                               attempts=1, status="verified", sources=sources)

        retry_prompt = build_prompt(
            q, concept_context, correction_note(derived, q.official_answer)
        )
        text, _ = _complete(retry_prompt)
        derived = parse_final_answer(text)
        verified = answers_match(derived, q.official_answer)
        return Explanation(
            text=text, derived_answer=derived, official_answer=q.official_answer,
            verified=verified, attempts=2,
            status="corrected" if verified else "unverified", sources=sources,
        )
    except GroqError as e:
        log.exception("Groq call failed for %s", q.qid)
        return Explanation(
            text=f"The language model call failed: {e}", derived_answer="",
            official_answer=q.official_answer, verified=False,
            status="error", sources=sources,
        )
