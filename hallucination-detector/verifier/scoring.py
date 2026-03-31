"""
verifier/scoring.py -- Hybrid Scoring System (Phase 4) Async
"""

from verifier.similarity import compute_similarity
from verifier.judge import judge_answer
from config import logger

WEIGHT_SIMILARITY = 0.4
WEIGHT_JUDGE      = 0.6

GATE_LOW  = 0.4
GATE_HIGH = 0.8

def _gate_label(similarity: float) -> str:
    if similarity < GATE_LOW:
        return "SKIP_JUDGE_LOW"
    if similarity >= GATE_HIGH:
        return "OPTIONAL_JUDGE"
    return "SEND_TO_JUDGE"

async def compute_hybrid_score(
    answer: str,
    context_chunks: list[str],
    sim_result: dict | None = None,
) -> dict:
    logger.info("computing_hybrid_score")
    if sim_result is None:
        sim_result = await compute_similarity(answer, context_chunks)

    similarity_score = sim_result["max_score"]
    gate             = _gate_label(similarity_score)

    judge_result  = None
    judge_was_run = False
    judge_score   = 0.0

    logger.info("hybrid_scoring_gating", gate=gate, similarity=similarity_score)

    if gate == "SKIP_JUDGE_LOW":
        judge_score = 0.0
    else:
        judge_result  = await judge_answer(answer, context_chunks)
        judge_was_run = True
        judge_score   = judge_result["judge_score"]

    final_score = round(
        (WEIGHT_SIMILARITY * similarity_score) + (WEIGHT_JUDGE * judge_score),
        4,
    )

    logger.info("hybrid_scoring_complete", final_score=final_score)

    return {
        "similarity":    sim_result,
        "judge":         judge_result,
        "final_score":   final_score,
        "gating_label":  gate,
        "judge_was_run": judge_was_run,
    }
