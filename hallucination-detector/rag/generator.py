"""
rag/generator.py — Answer generator using Gemini 2.5 Flash.

Takes a user question + retrieved context chunks and produces a
grounded answer via the Gemini 2.5 Flash model.
"""

from config import client, LLM_MODEL

# ── Prompt template ────────────────────────────────────────────────────────────
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


def generate_answer(question: str, context_chunks: list[str]) -> dict:
    """
    Generate a grounded answer using Gemini 2.5 Flash.

    Args:
        question:       The user's question.
        context_chunks: Retrieved text chunks from the vector store.

    Returns:
        {
            "question": str,
            "context":  list[str],
            "answer":   str,
        }
    """
    prompt = build_prompt(question, context_chunks)

    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
    )

    return {
        "question": question,
        "context":  context_chunks,
        "answer":   response.text.strip(),
    }


if __name__ == "__main__":
    # Minimal smoke test — add a few dummy chunks to try it out
    sample_chunks = [
        "Python is a high-level, interpreted programming language known for its "
        "simplicity and readability.",
        "Python supports multiple programming paradigms including procedural, "
        "object-oriented, and functional programming.",
    ]
    result = generate_answer("What is Python?", sample_chunks)
    print("Question:", result["question"])
    print("\nAnswer:", result["answer"])
