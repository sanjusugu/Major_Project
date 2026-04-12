"""
api/main.py -- FastAPI server for the Async RAG Hallucination Detection System.
"""

import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from config import logger, db, collection
from rag.ingestion import ingest
from rag.retriever import Retriever
from rag.generator import generate_answer
from verifier.similarity import compute_similarity
from verifier.scoring import compute_hybrid_score

_retriever: Retriever | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _retriever
    logger.info("application_startup")
    docs_dir = Path("data/docs")
    
    try:
        # Scan for all PDFs
        files = [str(p) for p in docs_dir.glob("*") if p.suffix == ".pdf"]
        
        if files:
            logger.info("checking_for_new_documents", count=len(files))
            # The refactored ingest() now automatically skips existing files internally
            summary = await ingest(files)
            if summary.get("ingested"):
                logger.info("new_documents_ingested", files=summary["ingested"])
            else:
                logger.info("no_new_documents_found")
        
        # Initialize retriever if we have any data at all
        count = collection.count_documents({})
        if count > 0:
            _retriever = Retriever()
            logger.info("retriever_loaded", chunk_count=count)
        else:
            logger.warning("no_data_in_mongodb")
            
    except Exception as e:
        logger.error("startup_failed", error=str(e))
    yield
    logger.info("application_shutdown")


app = FastAPI(
    title="Async RAG Hallucination Detector",
    description="Detects hallucinations using Gemini 2.5 Flash, Hybrid Search (Dense+Sparse), and FlashRank.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(..., example="What is RAG?")
    top_k:    int = Field(4, ge=1, le=20)


class SimilarityResult(BaseModel):
    max_score:    float
    mean_score:   float
    chunk_scores: list[float]
    label:        str


class JudgeResult(BaseModel):
    verdict:     str
    confidence:  float
    judge_score: float
    explanation: str


class HybridResult(BaseModel):
    final_score:   float
    gating_label:  str
    judge_was_run: bool


class AskResponse(BaseModel):
    question:   str
    answer:     str
    similarity: SimilarityResult
    judge:      JudgeResult | None
    hybrid:     HybridResult


@app.get("/health", tags=["System"])
def health():
    return {
        "status":       "ok",
        "index_ready":  _retriever is not None,
        "features": [
            "Async Execution",
            "Advanced Unstructured Parsing",
            "Hybrid Search (BM25 + MongoDB)",
            "Cross-Encoder Re-ranking",
            "LLM-as-a-Judge Evaluation"
        ],
    }


@app.post("/ingest", tags=["System"])
async def ingest_docs(background_tasks: BackgroundTasks):
    """Trigger background ingestion of all .pdf files from data/docs/."""
    docs_dir = Path("data/docs")
    files = [str(p) for p in docs_dir.glob("*") if p.suffix == ".pdf"]
    if not files:
        raise HTTPException(status_code=404, detail="No .pdf files found in data/docs/")
    
    async def task(file_paths):
        global _retriever
        logger.info("background_ingestion_started")
        await ingest(file_paths)
        _retriever = Retriever()
        logger.info("background_ingestion_complete")

    background_tasks.add_task(task, files)
    return {"message": f"Started background ingestion for {len(files)} file(s)."}


@app.post("/ask", response_model=AskResponse, tags=["Pipeline"])
async def ask(req: AskRequest):
    """Full asynchronous pipeline: Retrieve -> Generate -> Similarity -> Judge"""
    logger.info("ask_request_received", question=req.question)
    
    if _retriever is None:
        raise HTTPException(status_code=503, detail="Index not ready. Call POST /ingest first.")

    hits = await _retriever.retrieve(req.question, top_k=req.top_k)
    context_chunks = [h["chunk"] for h in hits]
    
    if not context_chunks:
        logger.warning("no_chunks_found_for_question")
    
    # Generate answer
    gen_result = await generate_answer(req.question, context_chunks)
    answer = gen_result["answer"]

    # Semantic similarity phase
    sim_result = await compute_similarity(answer, context_chunks)

    # Hybrid scoring phase (Judgement)
    hybrid = await compute_hybrid_score(answer, context_chunks, sim_result=sim_result)

    judge_out = None
    if hybrid["judge"] is not None:
        judge_out = JudgeResult(**hybrid["judge"])

    logger.info("ask_request_complete", final_score=hybrid["final_score"])
    
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

# MOUNT STATIC FILES AFTER ROUTES
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", include_in_schema=False)
async def serve_ui():
    return FileResponse(os.path.join("static", "index.html"))
