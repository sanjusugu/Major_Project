"""
verifier/judge.py -- Gemini LLM-as-a-Judge Module (Phase 3) Async
"""

import re
import json
from config import client, LLM_MODEL, logger

VERDICT_SCORES = {
    "SUPPORTED":     1.0,
    "PARTIAL":       0.6,
    "NOT_SUPPORTED": 0.0,
}

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

def _build_prompt(answer: str, context_chunks: list[str]) -> str:
    context_text = "\n\n---\n\n".join(context_chunks)
    return JUDGE_PROMPT_TEMPLATE.format(context=context_text, answer=answer)

def _parse_response(raw: str) -> dict:
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.error("judge_parse_failure", raw=raw)
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            return {
                "verdict":     "PARTIAL",
                "confidence":  0.5,
                "explanation": "Could not parse judge response.",
                "raw":         raw,
            }

    verdict    = str(data.get("verdict", "PARTIAL")).upper().strip()
    confidence = float(data.get("confidence", 0.5))
    explanation= str(data.get("explanation", ""))

    confidence = max(0.0, min(1.0, confidence))
    if verdict not in VERDICT_SCORES:
        verdict = "PARTIAL"

    return {
        "verdict":     verdict,
        "confidence":  round(confidence, 4),
        "explanation": explanation,
    }

async def judge_answer(answer: str, context_chunks: list[str]) -> dict:
    logger.info("starting_llm_judge")
    prompt   = _build_prompt(answer, context_chunks)
    
    response = await client.aio.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
    )

    parsed = _parse_response(response.text.strip())

    verdict_score = VERDICT_SCORES.get(parsed["verdict"], 0.6)
    judge_score   = round(verdict_score * parsed["confidence"], 4)

    logger.info("llm_judge_complete", verdict=parsed["verdict"], score=judge_score)

    return {
        "verdict":     parsed["verdict"],
        "confidence":  parsed["confidence"],
        "judge_score": judge_score,
        "explanation": parsed["explanation"],
    }
