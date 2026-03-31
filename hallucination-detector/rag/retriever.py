"""
rag/retriever.py — Advanced Hybrid Query Retriever using Dense + Sparse + CrossEncoder.

Loads the ChromaDB collection + BM25 index, retrieves top 10 from each,
deduplicates, and reranks down to top K using FlashRank (bge-reranker or similar).
"""

import os
import pickle
import chromadb

# Dense + Sparse
from config import client, EMBEDDING_MODEL, logger
from rag.ingestion import embed_texts, DB_PATH, BM25_PATH, COLLECTION_NAME

# Re-ranking
from flashrank import Ranker, RerankRequest


class Retriever:
    """Async Hybrid Retriever using ChromaDB + BM25 + FlashRank."""
    
    def __init__(self) -> None:
        logger.info("loading_retriever")
        
        self.chroma_client = chromadb.PersistentClient(path=DB_PATH)
        self.collection = self.chroma_client.get_collection(name=COLLECTION_NAME)
        
        # Load Sparse BM25
        if not os.path.exists(BM25_PATH):
            logger.error("bm25_not_found", path=BM25_PATH)
            raise ValueError("BM25 index not found. Run ingest first.")
            
        with open(BM25_PATH, "rb") as f:
            data = pickle.load(f)
            self.bm25 = data["bm25"]
            self.bm25_chunks = data["chunks"]
            
        # Initialize CrossEncoder Re-ranker
        logger.info("loading_ranker_model")
        # Use default model which is extremely fast and effective for semantic ranking
        self.ranker = Ranker(cache_dir="data/flashrank")
        logger.info("retriever_ready")

    async def retrieve(self, query: str, top_k: int = 4) -> list[dict]:
        """
        Retrieval Flow:
         1. BM25 Query (Sparse)
         2. Gemini Embed Query -> Chroma (Dense)
         3. Union & Deduplicate
         4. FlashRank Rerank
        """
        logger.info("starting_retrieval", query=query, top_k=top_k)
        
        # 1. Sparse Search
        tokenized_query = query.lower().split()
        sparse_scores = self.bm25.get_scores(tokenized_query)
        top_sparse_idx = sorted(range(len(sparse_scores)), key=lambda i: sparse_scores[i], reverse=True)[:10]
        sparse_chunks = [self.bm25_chunks[i] for i in top_sparse_idx]
        logger.info("sparse_retrieved", count=len(sparse_chunks))

        # 2. Dense Search
        query_vec = await embed_texts([query])
        results = self.collection.query(
            query_embeddings=query_vec.tolist(),
            n_results=10
        )
        dense_chunks = results['documents'][0] if results['documents'] else []
        logger.info("dense_retrieved", count=len(dense_chunks))
        
        # 3. Union + Deduplication
        unique_chunks = list(set(sparse_chunks + dense_chunks))
        logger.info("deduplicated_chunks", count=len(unique_chunks))
        
        if not unique_chunks:
            return []

        # 4. Re-ranker Pipeline
        passages = [{"id": i, "text": chunk} for i, chunk in enumerate(unique_chunks)]
        rerank_request = RerankRequest(query=query, passages=passages)
        
        ranked_results = self.ranker.rerank(rerank_request)
        logger.info("reranking_complete")
        
        final_hits = []
        # Return only the highly relevance-scored top K
        for res in ranked_results[:top_k]:
            final_hits.append({"chunk": res["text"], "score": res["score"]})
                
        return final_hits
