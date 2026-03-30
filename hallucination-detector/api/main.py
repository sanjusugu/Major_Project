"""
api/main.py -- FastAPI server for the RAG Hallucination Detection System.

Phases active:
  Phase 1 -- RAG (ingest, retrieve, generate)
  Phase 2 -- Semantic Similarity
  Phase 3 -- Gemini LLM-as-a-Judge
  Phase 4 -- Hybrid Scoring (gating + weighted final score)

Run:
    uvicorn api.main:app --reload --port 8000
"""

import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag.ingestion import ingest
from rag.retriever import Retriever
from rag.generator import generate_answer
from verifier.similarity import compute_similarity
from verifier.scoring import compute_hybrid_score

INDEX_PATH = "data/faiss.index"
_retriever: Retriever | None = None


# ── Startup ────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _retriever
    docs_dir = Path("data/docs")
    if not os.path.exists(INDEX_PATH):
        files = [str(p) for p in docs_dir.glob("*") if p.suffix in {".txt", ".pdf"}]
        if files:
            print("[*] Auto-ingesting documents on startup...")
            ingest(files)
    if os.path.exists(INDEX_PATH):
        _retriever = Retriever()
        print("[OK] Retriever loaded.")
    else:
        print("[!] No index found. Call POST /ingest first.")
    yield


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Hallucination Detector",
    description="Detects hallucinations using Gemini 2.5 Flash — Phase 1-4 active.",
    version="0.4.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ────────────────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str = Field(..., example="What is RAG?")
    top_k:    int = Field(4, ge=1, le=20)


class SimilarityResult(BaseModel):
    max_score:    float
    mean_score:   float
    chunk_scores: list[float]
    label:        str          # HIGH / MEDIUM / LOW


class JudgeResult(BaseModel):
    verdict:     str           # SUPPORTED / PARTIAL / NOT_SUPPORTED
    confidence:  float
    judge_score: float
    explanation: str


class HybridResult(BaseModel):
    final_score:   float
    gating_label:  str         # SKIP_JUDGE_LOW | SEND_TO_JUDGE | OPTIONAL_JUDGE
    judge_was_run: bool


class AskResponse(BaseModel):
    question:   str
    answer:     str
    similarity: SimilarityResult
    judge:      JudgeResult | None
    hybrid:     HybridResult
    phases:     str = "Phase 1 + Phase 2 + Phase 3 + Phase 4"


class IngestResponse(BaseModel):
    message:    str
    files_used: list[str]


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health():
    return {
        "status":       "ok",
        "index_ready":  os.path.exists(INDEX_PATH),
        "phases_active": [
            "Phase 1 - RAG",
            "Phase 2 - Similarity",
            "Phase 3 - Judge",
            "Phase 4 - Hybrid Scoring",
        ],
    }


@app.post("/ingest", response_model=IngestResponse, tags=["System"])
def ingest_docs():
    """Ingest all .txt/.pdf files from data/docs/ into the FAISS index."""
    global _retriever
    docs_dir = Path("data/docs")
    files = [str(p) for p in docs_dir.glob("*") if p.suffix in {".txt", ".pdf"}]
    if not files:
        raise HTTPException(status_code=404, detail="No .txt or .pdf files found in data/docs/")
    ingest(files)
    _retriever = Retriever()
    return {"message": f"Ingested {len(files)} file(s) successfully.", "files_used": files}


@app.post("/ask", response_model=AskResponse, tags=["Pipeline"])
def ask(req: AskRequest):
    """
    Full pipeline: RAG -> Similarity -> Judge (gated) -> Hybrid Score.

    Returns a structured response with the answer, similarity scores,
    judge verdict (if run), and the final hybrid hallucination score.
    """
    if _retriever is None:
        raise HTTPException(status_code=503, detail="Index not ready. Call POST /ingest first.")

    # Phase 1: Retrieve + Generate
    hits           = _retriever.retrieve(req.question, top_k=req.top_k)
    context_chunks = [h["chunk"] for h in hits]
    answer         = generate_answer(req.question, context_chunks)["answer"]

    # Phase 2: Similarity (computed first so we can pass it to the hybrid scorer)
    sim_result = compute_similarity(answer, context_chunks)

    # Phase 3 + 4: Hybrid scoring (gated judge + weighted combination)
    hybrid = compute_hybrid_score(answer, context_chunks, sim_result=sim_result)

    judge_out = None
    if hybrid["judge"] is not None:
        judge_out = JudgeResult(**hybrid["judge"])

    return AskResponse(
        question   = req.question,
        answer     = answer,
        similarity = SimilarityResult(**sim_result),
        judge      = judge_out,
        hybrid     = HybridResult(
            final_score   = hybrid["final_score"],
            gating_label  = hybrid["gating_label"],
            judge_was_run = hybrid["judge_was_run"],
        ),
    )
