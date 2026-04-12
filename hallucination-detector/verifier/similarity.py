"""
verifier/similarity.py -- Semantic Similarity Module (Phase 2) Async
"""

import numpy as np
from config import client, embed_model, logger

THRESHOLDS = {
    "high":   0.8,
    "medium": 0.5,
}

async def _embed_async(texts: list[str]) -> np.ndarray:
    logger.info("embedding_batch_local", size=len(texts))
    vectors = embed_model.encode(texts)
    vectors = np.array(vectors, dtype=np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors

def score_label(score: float) -> str:
    if score >= THRESHOLDS["high"]:
        return "HIGH"
    if score >= THRESHOLDS["medium"]:
        return "MEDIUM"
    return "LOW"

async def compute_similarity(answer: str, context_chunks: list[str]) -> dict:
    logger.info("computing_similarity")
    if not context_chunks:
        return {
            "max_score":    0.0,
            "mean_score":   0.0,
            "chunk_scores": [],
            "label":        "LOW",
        }

    all_texts  = [answer] + context_chunks
    embeddings = await _embed_async(all_texts)

    answer_vec  = embeddings[0:1]
    chunk_vecs  = embeddings[1:]

    chunk_scores = (chunk_vecs @ answer_vec.T).flatten().tolist()

    max_score  = float(max(chunk_scores))
    mean_score = float(np.mean(chunk_scores))

    logger.info("similarity_computed", max_score=max_score, label=score_label(max_score))

    return {
        "max_score":    round(max_score,  4),
        "mean_score":   round(mean_score, 4),
        "chunk_scores": [round(s, 4) for s in chunk_scores],
        "label":        score_label(max_score),
    }
