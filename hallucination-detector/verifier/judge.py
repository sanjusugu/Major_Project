"""
verifier/judge.py -- Gemini LLM-as-a-Judge Module (Phase 3)

Uses Gemini 2.5 Flash to verify whether the generated answer is supported
by the retrieved context, outputting a structured verdict + confidence score.

Verdict options:
  SUPPORTED     -- answer is clearly grounded in the context
  PARTIAL       -- answer is partially grounded; some claims may be unsupported
  NOT_SUPPORTED -- answer contradicts or goes beyond the context
"""

import re
import json
from config import client, LLM_MODEL


# ── Score mapping ──────────────────────────────────────────────────────────────
VERDICT_SCORES = {
    "SUPPORTED":     1.0,
    "PARTIAL":       0.6,
    "NOT_SUPPORTED": 0.0,
}

# ── Judge prompt ───────────────────────────────────────────────────────────────
JUDGE_PROMPT_TEMPLATE = """You are an impartial fact-checking judge.

Your task is to evaluate whether the given ANSWER is supported by the provided CONTEXT.

CONTEXT:
{context}

ANSWER:
{answer}

Instructions:
- Read the context carefully.
- Check every claim in the answer against the context.
- Return your evaluation as valid JSON with exactly these fields:

{{
  "verdict": "SUPPORTED" | "PARTIAL" | "NOT_SUPPORTED",
  "confidence": <float between 0.0 and 1.0>,
  "explanation": "<one or two sentence justification>"
}}

Rules:
- Use SUPPORTED if the answer is fully grounded in the context.
- Use PARTIAL if the answer is only partly grounded or mixes supported and unsupported claims.
- Use NOT_SUPPORTED if the answer contradicts or goes beyond the context.
- confidence reflects how certain you are of your verdict (1.0 = very certain).
- Output ONLY the JSON object. No markdown, no extra text.
"""


# ── Helpers ────────────────────────────────────────────────────────────────────
def _build_prompt(answer: str, context_chunks: list[str]) -> str:
    context_text = "\n\n---\n\n".join(context_chunks)
    return JUDGE_PROMPT_TEMPLATE.format(context=context_text, answer=answer)


def _parse_response(raw: str) -> dict:
    """
    Extract and parse the JSON from the model's response.
    Handles cases where the model wraps the JSON in markdown code fences.
    """
    # Strip markdown fences if present
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: try to locate the first {...} block
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            # Cannot parse -- return a safe default
            return {
                "verdict":     "PARTIAL",
                "confidence":  0.5,
                "explanation": "Could not parse judge response.",
                "raw":         raw,
            }

    verdict    = str(data.get("verdict", "PARTIAL")).upper().strip()
    confidence = float(data.get("confidence", 0.5))
    explanation= str(data.get("explanation", ""))

    # Clamp confidence to [0, 1]
    confidence = max(0.0, min(1.0, confidence))

    # Validate verdict
    if verdict not in VERDICT_SCORES:
        verdict = "PARTIAL"

    return {
        "verdict":     verdict,
        "confidence":  round(confidence, 4),
        "explanation": explanation,
    }


# ── Main function ──────────────────────────────────────────────────────────────
def judge_answer(answer: str, context_chunks: list[str]) -> dict:
    """
    Use Gemini 2.5 Flash to judge if the answer is supported by the context.

    Args:
        answer:         The LLM-generated answer string.
        context_chunks: List of retrieved text chunks used as context.

    Returns:
        {
            "verdict":      str,    # SUPPORTED / PARTIAL / NOT_SUPPORTED
            "confidence":   float,  # 0.0 - 1.0
            "judge_score":  float,  # verdict_score * confidence
            "explanation":  str,
        }
    """
    prompt   = _build_prompt(answer, context_chunks)
    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
    )

    parsed = _parse_response(response.text.strip())

    verdict_score = VERDICT_SCORES.get(parsed["verdict"], 0.6)
    judge_score   = round(verdict_score * parsed["confidence"], 4)

    return {
        "verdict":     parsed["verdict"],
        "confidence":  parsed["confidence"],
        "judge_score": judge_score,
        "explanation": parsed["explanation"],
    }


# ── Quick standalone test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    answer = (
        "RAG reduces hallucinations by constraining the LLM to the provided "
        "context, ensuring answers are grounded in retrieved documents."
    )
    chunks = [
        "Benefits of RAG: Reduces hallucinations since the model is constrained "
        "to available context. Enables use of up-to-date or private knowledge.",
        "Retrieval-Augmented Generation (RAG) combines information retrieval "
        "with LLMs to produce more accurate and grounded answers.",
    ]

    result = judge_answer(answer, chunks)

    print("Judge Report")
    print("=" * 40)
    print(f"  Verdict     : {result['verdict']}")
    print(f"  Confidence  : {result['confidence']}")
    print(f"  Judge Score : {result['judge_score']}")
    print(f"  Explanation : {result['explanation']}")
