"""
rag/generator.py — Answer generator using Gemini 2.5 Flash (Async).
"""

from config import client, LLM_MODEL, logger

SYSTEM_PROMPT = """You are a helpful assistant that answers questions strictly
based on the provided context. If the context does not contain enough information
to answer the question, say "I don't know based on the given context."
Do NOT add information from outside the context."""


def build_prompt(question: str, context_chunks: list[str]) -> str:
    context_text = "\n\n---\n\n".join(context_chunks)
    return f"""{SYSTEM_PROMPT}

## Context:
{context_text}

## Question:
{question}

## Answer:"""


async def generate_answer(question: str, context_chunks: list[str]) -> dict:
    """Generate a grounded answer asynchronously using Gemini."""
    prompt = build_prompt(question, context_chunks)
    
    logger.info("generating_answer_start", chunks_used=len(context_chunks))

    response = await client.aio.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
    )
    
    logger.info("generating_answer_complete")

    return {
        "question": question,
        "context":  context_chunks,
        "answer":   response.text.strip(),
    }
