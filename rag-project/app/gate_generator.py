"""
Generates a step-by-step explanation for a GATE question, then verifies
the explanation's derived answer against the official answer key before
it's shown to the student. This is the piece that makes the project more
than a wrapper around an LLM: a wrong derivation is worse than no answer,
so nothing is shown unverified without a visible warning.

Flow:
  1. Generate an explanation grounded in the question + retrieved concept
     context, ending with a machine-parseable "FINAL ANSWER: X" line.
  2. Parse the model's final answer out of that line.
  3. Compare against the official answer key (already ground truth from
     the parsed answer-key CSV -- no LLM involved in that comparison).
  4. If they disagree, retry once with an explicit correction instruction
     naming the official answer. If they still disagree, the explanation
     is shown with a visible "unverified" flag rather than silently
     trusted -- this is the abstention behavior for this domain.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from groq import Groq

from app.config import GENERATION_MODEL, GROQ_API_KEY
from app.gate_parser import GateQuestion

FINAL_ANSWER_RE = re.compile(r"FINAL ANSWER:\s*(.+)", re.IGNORECASE)

SYSTEM_PROMPT = """You are a GATE exam tutor. Given a question, its type \
(MCQ/MSQ/NAT), and optional concept reference material, produce a clear, \
step-by-step derivation a student can follow. Rules:
- Work through the reasoning in numbered steps, referencing concepts by name.
- Do not skip steps a student would need to reproduce the logic themselves.
- End with a line in exactly this format: "FINAL ANSWER: <answer>"
  - For MCQ: a single letter, e.g. "FINAL ANSWER: B"
  - For MSQ: comma-separated letters, e.g. "FINAL ANSWER: B,C,D"
  - For NAT: the numeric value, e.g. "FINAL ANSWER: 4096"
"""


@dataclass
class Explanation:
    text: str
    derived_answer: str
    official_answer: str
    verified: bool


def _build_prompt(q: GateQuestion, concept_context: str = "", correction_note: str = "") -> str:
    options_block = "\n".join(f"({letter}) {text}" for letter, text in q.options.items())
    parts = [
        f"Question type: {q.q_type}, marks: {q.marks}",
        f"Question: {q.stem}",
    ]
    if options_block:
        parts.append(f"Options:\n{options_block}")
    if concept_context:
        parts.append(f"Relevant concept reference:\n{concept_context}")
    if correction_note:
        parts.append(correction_note)
    parts.append("Explain step by step, then give the FINAL ANSWER line.")
    return "\n\n".join(parts)


def _parse_final_answer(text: str) -> str:
    m = FINAL_ANSWER_RE.search(text)
    return m.group(1).strip().upper().replace(" ", "") if m else ""


def _answers_match(derived: str, official: str) -> bool:
    """Official NAT answers are stored as ranges like '2.374 to 2.376' or
    '2 to 2 OR 4 to 4'; MSQ answers are comma-separated letter sets where
    order shouldn't matter."""
    official = official.strip().upper()
    if "TO" in official:
        try:
            ranges = [r.strip() for r in official.split(" OR ")]
            value = float(derived)
            for r in ranges:
                lo, hi = [float(x.strip()) for x in r.split("TO")]
                if lo <= value <= hi:
                    return True
            return False
        except ValueError:
            return derived == official
    if "," in official or "," in derived:
        return set(official.split(",")) == set(derived.split(","))
    return derived == official


def _call_model(prompt: str) -> str:
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content or ""


def generate_verified_explanation(q: GateQuestion, concept_context: str = "") -> Explanation:
    if not GROQ_API_KEY:
        return Explanation(
            text="[No GROQ_API_KEY set -- generation skipped.]",
            derived_answer="", official_answer=q.official_answer, verified=False,
        )

    prompt = _build_prompt(q, concept_context)
    text = _call_model(prompt)
    derived = _parse_final_answer(text)
    verified = _answers_match(derived, q.official_answer)

    if not verified:
        correction = (
            f"Your previous derivation gave FINAL ANSWER: {derived}, but the "
            f"verified official answer is {q.official_answer}. Re-derive the "
            f"solution so the reasoning actually arrives at the official answer. "
            f"If you believe the official answer is wrong, say so explicitly "
            f"instead of forcing an incorrect derivation."
        )
        retry_prompt = _build_prompt(q, concept_context, correction_note=correction)
        text = _call_model(retry_prompt)
        derived = _parse_final_answer(text)
        verified = _answers_match(derived, q.official_answer)

    return Explanation(
        text=text, derived_answer=derived,
        official_answer=q.official_answer, verified=verified,
    )
