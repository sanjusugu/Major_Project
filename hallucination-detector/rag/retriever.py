"""
rag/retriever.py — Query retriever.

Loads the FAISS index from disk, embeds the user query using the Gemini
embedding model and returns the top-k most similar text chunks.
"""

import numpy as np
from config import client, EMBEDDING_MODEL
from rag.ingestion import load_store, embed_texts


class Retriever:
    def __init__(self) -> None:
        self.index, self.chunks = load_store()

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Embed the query and find the top-k closest chunks.

        Returns a list of dicts:
          { "chunk": str, "score": float }
        """
        # Embed and normalise (must match ingestion normalisation)
        query_vec = embed_texts([query])          # shape (1, D)
        import faiss
        faiss.normalize_L2(query_vec)

        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({"chunk": self.chunks[idx], "score": float(score)})

        return results


if __name__ == "__main__":
    retriever = Retriever()
    query = "What is retrieval-augmented generation?"
    hits  = retriever.retrieve(query, top_k=3)
    for i, hit in enumerate(hits, 1):
        print(f"\n--- Result {i} (score={hit['score']:.4f}) ---")
        print(hit["chunk"][:300])
