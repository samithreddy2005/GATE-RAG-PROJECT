"""
FastAPI surface for the GATE prep assistant.

Endpoint groups:
  browse    /api/subjects, /api/topics, /api/questions   -- structured SQL
  practice  /api/practice, /api/attempt                  -- answers withheld
                                                            until submitted
  explain   /api/explain, /api/explain/stream            -- verified against
                                                            the answer key
  ask       /api/chat                                    -- free-form RAG
  admin     /api/status, /api/ingest, /api/rebuild

Handlers that touch the index are plain `def`, not `async def`. FastAPI runs
those in a threadpool, so the blocking SQLite, embedding and HTTP work cannot
stall the event loop -- which an `async def` doing the same work silently
would.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
from contextlib import asynccontextmanager
from typing import Iterator, Literal

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.build_index import Corpus, build_corpus
from app.config import (
    CONFIDENCE_THRESHOLD,
    FRONTEND_DIR,
    GENERATION_MODEL,
    GROQ_API_KEY,
    TOP_K_FINAL,
    UPLOAD_DIR,
)
from app.gate_generator import (
    Explanation,
    answers_match,
    build_concept_context,
    generate_verified_explanation,
    stream_tokens,
    build_prompt,
    correction_note,
    parse_final_answer,
)
from app.gate_parser import GateQuestion
from app.generator import stream_answer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("gate-rag")

_corpus: Corpus | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the index once at startup. A failure here is logged rather than
    fatal so the server still comes up and /api/status can explain what is
    wrong -- a dead port is a much worse debugging experience than a running
    server reporting "no documents indexed"."""
    global _corpus
    try:
        _corpus = build_corpus()
        log.info("Indexed %d questions and %d concept chunks from %s",
                 _corpus.n_questions, _corpus.n_concept_chunks,
                 ", ".join(_corpus.papers) or "no papers")
    except Exception:
        log.exception("Startup indexing failed")
        _corpus = None
    yield
    if _corpus is not None:
        _corpus.bank.close()


app = FastAPI(
    title="GATE RAG Assistant",
    description="Past-paper retrieval with answer-key-verified explanations.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


def get_corpus() -> Corpus:
    if _corpus is None:
        raise HTTPException(
            status_code=503,
            detail="Index unavailable. Run `python -m app.build_index` and check the server log.",
        )
    return _corpus


# --------------------------------------------------------------- serializers

def question_public(q: GateQuestion, reveal_answer: bool = True) -> dict:
    """`reveal_answer=False` is what makes practice mode honest: the key is
    never sent to the browser before the student has committed to an answer,
    so it cannot be read out of devtools or the network tab."""
    payload = {
        "qid": q.qid,
        "subject": q.subject,
        "section": q.section,
        "year": q.year,
        "q_no": q.q_no,
        "marks": q.marks,
        "q_type": q.q_type,
        "stem": q.stem,
        "options": q.options,
        "topics": q.topics,
    }
    if reveal_answer:
        payload["official_answer"] = q.official_answer
        payload["is_answerable"] = q.is_answerable
    return payload


def explanation_public(e: Explanation) -> dict:
    return {
        "text": e.text,
        "derived_answer": e.derived_answer,
        "official_answer": e.official_answer,
        "verified": e.verified,
        "status": e.status,
        "attempts": e.attempts,
        "note": e.student_note,
        "sources": e.sources,
    }


def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


# --------------------------------------------------------------------- admin

@app.get("/api/status")
def status():
    if _corpus is None:
        return {
            "ready": False,
            "detail": "Index not built. See server log; then POST /api/rebuild.",
            "generation_configured": bool(GROQ_API_KEY),
        }
    return {
        "ready": True,
        "generation_configured": bool(GROQ_API_KEY),
        "generation_model": GENERATION_MODEL,
        "indexed_questions": _corpus.n_questions,
        "indexed_concept_chunks": _corpus.n_concept_chunks,
        "vector_store_size": _corpus.vector_store.count(),
        "papers": _corpus.papers,
        "bank": _corpus.bank.summary(),
    }


@app.post("/api/rebuild")
def rebuild():
    global _corpus
    _corpus = build_corpus()
    return {"status": "rebuilt", "questions": _corpus.n_questions,
            "concept_chunks": _corpus.n_concept_chunks}


SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]")


def _safe_filename(name: str) -> str:
    """Uploads are written to disk, so the client-supplied name is untrusted.
    Taking only the basename and stripping everything outside a strict
    allowlist blocks `../` traversal and absolute paths on both POSIX and
    Windows."""
    base = name.replace("\\", "/").split("/")[-1]
    cleaned = SAFE_NAME_RE.sub("_", base).lstrip(".")
    return cleaned or "upload.txt"


@app.post("/api/ingest")
def ingest(files: list[UploadFile] = File(...)):
    """Add concept/reference notes (.txt, .md, .pdf) to the corpus."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for f in files:
        name = _safe_filename(f.filename or "upload.txt")
        if not name.lower().endswith((".txt", ".md", ".pdf")):
            raise HTTPException(400, f"Unsupported file type: {f.filename}")
        with (UPLOAD_DIR / name).open("wb") as out:
            shutil.copyfileobj(f.file, out)
        written.append(name)

    global _corpus
    _corpus = build_corpus()
    return {"status": "indexed", "files": written,
            "concept_chunks": _corpus.n_concept_chunks}


# -------------------------------------------------------------------- browse

@app.get("/api/subjects")
def subjects(corpus: Corpus = Depends(get_corpus)):
    return {"subjects": corpus.bank.subjects()}


@app.get("/api/topics")
def topics(subject: str | None = None, corpus: Corpus = Depends(get_corpus)):
    """Frequency and mark-weight per topic -- the "what should I revise"
    view. Mark weight is the more actionable of the two, since GATE scores
    by marks and not by question count."""
    frequency = corpus.bank.topic_frequency(subject=subject)
    marks = corpus.bank.topic_marks(subject=subject)
    total_marks = sum(marks.values()) or 1
    return {
        "subject": subject,
        "topics": [
            {
                "topic": topic,
                "questions": count,
                "marks": marks.get(topic, 0),
                "marks_share": round(100 * marks.get(topic, 0) / total_marks, 1),
            }
            for topic, count in frequency.items()
        ],
    }


@app.get("/api/questions")
def questions(
    subject: str | None = None,
    topic: str | None = None,
    q_type: Literal["MCQ", "MSQ", "NAT"] | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    limit: int = Query(50, ge=1, le=500),
    corpus: Corpus = Depends(get_corpus),
):
    rows = corpus.bank.filter(subject=subject, topic=topic, q_type=q_type,
                              year_min=year_min, year_max=year_max, limit=limit)
    return {"count": len(rows), "questions": [question_public(q) for q in rows]}


@app.get("/api/questions/{qid}")
def question_detail(qid: str, corpus: Corpus = Depends(get_corpus)):
    q = corpus.bank.get(qid)
    if q is None:
        raise HTTPException(404, f"No question with id {qid}")
    return question_public(q)


# ------------------------------------------------------------------ practice

@app.get("/api/practice")
def practice(
    subject: str | None = None,
    topic: str | None = None,
    q_type: Literal["MCQ", "MSQ", "NAT"] | None = None,
    n: int = Query(5, ge=1, le=50),
    seed: int | None = None,
    corpus: Corpus = Depends(get_corpus),
):
    """A random practice set with answers withheld. `answerable_only` drops
    questions GATE annulled, which have no correct option to grade against."""
    picked = corpus.bank.sample(
        n=n, seed=seed, subject=subject, topic=topic, q_type=q_type,
        answerable_only=True,
    )
    return {
        "count": len(picked),
        "questions": [question_public(q, reveal_answer=False) for q in picked],
    }


class AttemptRequest(BaseModel):
    qid: str
    answer: str = Field(..., description="Letter(s) for MCQ/MSQ, or a number for NAT")


@app.post("/api/attempt")
def attempt(req: AttemptRequest, corpus: Corpus = Depends(get_corpus)):
    """Grade a submitted answer against the official key.

    Grading is the same deterministic comparison used to verify generated
    explanations -- no model is consulted, so a student's score is never at
    the mercy of an LLM's opinion. GATE's own marking rules apply: MCQ carries
    negative marking, MSQ and NAT do not.
    """
    q = corpus.bank.get(req.qid)
    if q is None:
        raise HTTPException(404, f"No question with id {req.qid}")
    if not q.is_answerable:
        return {"qid": q.qid, "graded": False, "reason": "no_official_answer",
                "official_answer": q.official_answer, "marks_awarded": q.marks}

    submitted = req.answer.strip().upper().replace(" ", "")
    correct = answers_match(submitted, q.official_answer)
    if correct:
        awarded = float(q.marks)
    elif q.q_type == "MCQ":
        awarded = -(1 / 3) * q.marks   # GATE MCQ negative marking
    else:
        awarded = 0.0

    return {
        "qid": q.qid,
        "graded": True,
        "submitted": submitted,
        "correct": correct,
        "official_answer": q.official_answer,
        "marks": q.marks,
        "marks_awarded": round(awarded, 3),
        "topics": q.topics,
    }


# ------------------------------------------------------------------- explain

def _retrieve_for(q: GateQuestion, corpus: Corpus, top_k: int):
    """Retrieve concept passages for a question, deliberately excluding
    other questions. Explaining a question by quoting a different exam
    question teaches nothing; the "why" has to come from the concept notes."""
    query = f"{' '.join(q.topics)} {q.stem}"
    return corpus.retriever.retrieve(query, top_k=top_k, where={"kind": "concept"})


class ExplainRequest(BaseModel):
    qid: str
    top_k: int = Field(TOP_K_FINAL, ge=1, le=10)


@app.post("/api/explain")
def explain(req: ExplainRequest, corpus: Corpus = Depends(get_corpus)):
    q = corpus.bank.get(req.qid)
    if q is None:
        raise HTTPException(404, f"No question with id {req.qid}")
    chunks = _retrieve_for(q, corpus, req.top_k)
    return {
        "question": question_public(q),
        "explanation": explanation_public(generate_verified_explanation(q, chunks)),
    }


@app.post("/api/explain/stream")
def explain_stream(req: ExplainRequest, corpus: Corpus = Depends(get_corpus)):
    """Streams the derivation as it is written, then runs verification and
    emits the verdict. If the first attempt disagrees with the answer key the
    correction pass is streamed too, so the student watches the verification
    loop work rather than being handed an opaque badge."""
    q = corpus.bank.get(req.qid)
    if q is None:
        raise HTTPException(404, f"No question with id {req.qid}")

    chunks = _retrieve_for(q, corpus, req.top_k)
    concept_context, sources = build_concept_context(chunks)

    def event_stream() -> Iterator[str]:
        yield sse({"type": "question", "question": question_public(q, reveal_answer=False)})
        yield sse({"type": "sources", "sources": sources})

        if not GROQ_API_KEY:
            yield sse({"type": "error", "text": "No GROQ_API_KEY configured. "
                                                "Add one to rag-project/.env and restart."})
            return
        if not q.is_answerable:
            yield sse({"type": "verdict", "status": "no_key", "verified": False,
                       "official_answer": q.official_answer,
                       "note": "GATE published no single correct answer for this question."})
            return

        try:
            for attempt_no in (1, 2):
                if attempt_no == 1:
                    prompt = build_prompt(q, concept_context)
                else:
                    yield sse({"type": "retry",
                               "text": "Derivation disagreed with the official key. "
                                       "Re-deriving with the key as a constraint."})
                    prompt = build_prompt(
                        q, concept_context, correction_note(derived, q.official_answer)
                    )

                yield sse({"type": "attempt", "attempt": attempt_no})
                text = ""
                for token in stream_tokens(prompt):
                    text += token
                    yield sse({"type": "token", "text": token})

                derived = parse_final_answer(text)
                if answers_match(derived, q.official_answer):
                    yield sse({
                        "type": "verdict", "verified": True,
                        "status": "verified" if attempt_no == 1 else "corrected",
                        "derived_answer": derived,
                        "official_answer": q.official_answer,
                        "attempts": attempt_no,
                    })
                    return

            yield sse({
                "type": "verdict", "verified": False, "status": "unverified",
                "derived_answer": derived, "official_answer": q.official_answer,
                "attempts": 2,
                "note": "This derivation does not match the official answer key. "
                        "Treat it as a hint, not a solution.",
            })
        except Exception as e:                       # noqa: BLE001
            log.exception("Explanation stream failed for %s", q.qid)
            yield sse({"type": "error", "text": f"Generation failed: {e}"})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ----------------------------------------------------------------------- ask

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(TOP_K_FINAL, ge=1, le=10)
    subject: str | None = None


@app.post("/api/chat")
def chat(req: ChatRequest, corpus: Corpus = Depends(get_corpus)):
    where = {"subject": req.subject} if req.subject else None
    chunks = corpus.retriever.retrieve(req.question, top_k=req.top_k, where=where)

    def event_stream() -> Iterator[str]:
        yield sse({
            "type": "sources",
            "sources": [
                {"index": i + 1, "source": c.metadata.get("source") or c.source,
                 "kind": c.kind, "score": c.score, "qid": c.metadata.get("qid"),
                 "preview": c.text[:200]}
                for i, c in enumerate(chunks)
            ],
        })
        if not chunks:
            yield sse({"type": "low_confidence", "threshold": CONFIDENCE_THRESHOLD})
        for token in stream_answer(req.question, chunks):
            yield sse({"type": "token", "text": token})
        yield sse({"type": "done"})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# Serves the UI from this same server at http://127.0.0.1:8000/ .
# Mounted last so every /api route above takes precedence over the catch-all.
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
