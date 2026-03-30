"""
verifier/similarity.py -- Semantic Similarity Module (Phase 2)

Measures how well the generated answer is supported by the retrieved context
using cosine similarity between Gemini embeddings.

Steps:
  1. Embed the generated answer
  2. Embed each context chunk
  3. Compute cosine similarity (answer vs each chunk)
  4. Return MAX similarity score + a human-readable label
"""

import numpy as np
import faiss
from config import client, EMBEDDING_MODEL


# ── Similarity thresholds ──────────────────────────────────────────────────────
THRESHOLDS = {
    "high":   0.8,   # > 0.8  -> HIGH
    "medium": 0.5,   # 0.5-0.8 -> MEDIUM
                     # < 0.5  -> LOW
}


def _embed(texts: list[str]) -> np.ndarray:
    """Embed a list of texts and return a normalised float32 array (N, D)."""
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
    )
    vectors = np.array([e.values for e in response.embeddings], dtype=np.float32)
    faiss.normalize_L2(vectors)   # in-place L2 normalise -> cosine via dot product
    return vectors


def score_label(score: float) -> str:
    """Map a cosine similarity score to a human-readable label."""
    if score >= THRESHOLDS["high"]:
        return "HIGH"
    if score >= THRESHOLDS["medium"]:
        return "MEDIUM"
    return "LOW"


def compute_similarity(answer: str, context_chunks: list[str]) -> dict:
    """
    Compute semantic similarity between the answer and each context chunk.

    Args:
        answer:         The LLM-generated answer string.
        context_chunks: List of retrieved text chunks used as context.

    Returns:
        {
            "max_score":       float,   # highest cosine similarity (0-1)
            "mean_score":      float,   # average across all chunks
            "chunk_scores":    list[float],
            "label":           str,     # HIGH / MEDIUM / LOW
        }
    """
    if not context_chunks:
        return {
            "max_score":    0.0,
            "mean_score":   0.0,
            "chunk_scores": [],
            "label":        "LOW",
        }

    # Embed answer + all chunks in a single API call (efficient)
    all_texts  = [answer] + context_chunks
    embeddings = _embed(all_texts)

    answer_vec  = embeddings[0:1]            # shape (1, D)
    chunk_vecs  = embeddings[1:]             # shape (N, D)

    # Cosine similarity = dot product after L2 normalisation
    chunk_scores = (chunk_vecs @ answer_vec.T).flatten().tolist()  # (N,)

    max_score  = float(max(chunk_scores))
    mean_score = float(np.mean(chunk_scores))

    return {
        "max_score":    round(max_score,  4),
        "mean_score":   round(mean_score, 4),
        "chunk_scores": [round(s, 4) for s in chunk_scores],
        "label":        score_label(max_score),
    }


# ── Quick standalone test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    answer = (
        "RAG reduces hallucinations by constraining the LLM to the provided "
        "context, ensuring answers are grounded in retrieved documents."
    )
    chunks = [
        "Retrieval-Augmented Generation (RAG) combines information retrieval "
        "with LLMs to produce more accurate and grounded answers.",
        "Benefits of RAG: Reduces hallucinations since the model is constrained "
        "to available context.",
        "FAISS is an open-source library for efficient similarity search over "
        "dense vector collections.",
        "Python is a general-purpose programming language.",   # low-relevance chunk
    ]

    result = compute_similarity(answer, chunks)

    print("Similarity Report")
    print("=" * 40)
    print(f"  Max Score  : {result['max_score']}  ({result['label']})")
    print(f"  Mean Score : {result['mean_score']}")
    print(f"  Per-chunk  : {result['chunk_scores']}")
