"""
verifier/scoring.py -- Hybrid Scoring System (Phase 4)

Combines:
  - Semantic similarity score (Phase 2) -- fast, cheap signal
  - Gemini LLM judge score  (Phase 3) -- deep, reasoning signal

Gating logic avoids unnecessary judge calls:
  similarity < 0.4  --> skip judge, flag as hallucination
  similarity > 0.8  --> judge optional (still runs for accuracy)
  0.4 - 0.8         --> always run judge

Final score formula:
  final_score = (0.4 * similarity) + (0.6 * judge_score)
"""

from verifier.similarity import compute_similarity
from verifier.judge import judge_answer


# ── Weights & gates ────────────────────────────────────────────────────────────
WEIGHT_SIMILARITY = 0.4
WEIGHT_JUDGE      = 0.6

GATE_LOW  = 0.4   # below this: skip judge, certainhallucination
GATE_HIGH = 0.8   # above this: very likely safe, judge still runs


def _gate_label(similarity: float) -> str:
    if similarity < GATE_LOW:
        return "SKIP_JUDGE_LOW"     # definitely low -- skip judge
    if similarity >= GATE_HIGH:
        return "OPTIONAL_JUDGE"     # high similarity -- judge still runs
    return "SEND_TO_JUDGE"          # ambiguous -- must judge


def compute_hybrid_score(
    answer: str,
    context_chunks: list[str],
    sim_result: dict | None = None,
) -> dict:
    """
    Run the hybrid scoring pipeline.

    Args:
        answer:         Generated answer string.
        context_chunks: Retrieved context chunks.
        sim_result:     Optional pre-computed similarity dict (Phase 2).
                        If None, similarity is computed here.

    Returns:
        {
            "similarity":     dict,   # Phase 2 result
            "judge":          dict | None,  # Phase 3 result (None if gated out)
            "final_score":    float,  # weighted combination
            "gating_label":   str,    # SKIP_JUDGE_LOW | SEND_TO_JUDGE | OPTIONAL_JUDGE
            "judge_was_run":  bool,
        }
    """
    # ── Step 1: Similarity ─────────────────────────────────────────────────────
    if sim_result is None:
        sim_result = compute_similarity(answer, context_chunks)

    similarity_score = sim_result["max_score"]
    gate             = _gate_label(similarity_score)

    # ── Step 2: Gating ─────────────────────────────────────────────────────────
    judge_result  = None
    judge_was_run = False
    judge_score   = 0.0

    if gate == "SKIP_JUDGE_LOW":
        # Similarity is too low to bother with the judge.
        # Treat as NOT_SUPPORTED with 0 judge score.
        judge_score = 0.0

    else:
        # Run judge for SEND_TO_JUDGE and OPTIONAL_JUDGE
        judge_result  = judge_answer(answer, context_chunks)
        judge_was_run = True
        judge_score   = judge_result["judge_score"]

    # ── Step 3: Final weighted score ───────────────────────────────────────────
    final_score = round(
        (WEIGHT_SIMILARITY * similarity_score) + (WEIGHT_JUDGE * judge_score),
        4,
    )

    return {
        "similarity":    sim_result,
        "judge":         judge_result,
        "final_score":   final_score,
        "gating_label":  gate,
        "judge_was_run": judge_was_run,
    }


# ── Quick standalone test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    answer = (
        "RAG reduces hallucinations by grounding the LLM in retrieved context."
    )
    chunks = [
        "Benefits of RAG: Reduces hallucinations since the model is constrained "
        "to available context.",
        "Retrieval-Augmented Generation combines retrieval with language models.",
    ]

    result = compute_hybrid_score(answer, chunks)

    sim   = result["similarity"]
    judge = result["judge"]

    print("Hybrid Scoring Report (Phase 4)")
    print("=" * 45)
    print(f"  Similarity Score : {sim['max_score']}  ({sim['label']})")
    print(f"  Gating           : {result['gating_label']}")
    print(f"  Judge was run    : {result['judge_was_run']}")
    if judge:
        print(f"  Judge Verdict    : {judge['verdict']}  (conf={judge['confidence']})")
        print(f"  Judge Score      : {judge['judge_score']}")
    print(f"  Final Score      : {result['final_score']}")
    print("=" * 45)
