"""
rag/ingestion.py -- Advanced Document ingestion pipeline using ChromaDB + BM25 Sparse Search + Unstructured.

Steps:
  1. Load text from .pdf files using unstructured's semantic chunking
  2. Emit async embeddings via Gemini
  3. Store in a ChromaDB index for sparse search
  4. Store in BM25 index for dense search
"""

import os
import chromadb
import numpy as np
import pickle
from pathlib import Path

# Advanced Parser Imports
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

from rank_bm25 import BM25Okapi
from config import client, EMBEDDING_MODEL, logger

# Paths
DB_PATH = "data/chroma_db"
BM25_PATH = "data/bm25.pkl"
COLLECTION_NAME = "rag_chunks"


# 1. Advanced Loaders and Semantic Chunkers
def load_and_chunk_pdf(path: str) -> list[str]:
    """Uses unstructured layout-aware parsing and chunking by semantic title headers."""
    logger.info("partitioning_pdf", path=path)
    
    # We partition directly into elements
    # Using 'fast' strategy to ensure it works smoothly without OCR heavy dependencies,
    # but still respecting layout elements.
    elements = partition_pdf(path, strategy="fast")
    
    # Chunk semantically
    chunks = chunk_by_title(
        elements, 
        max_characters=1000,
        combine_text_under_n_chars=400,
        new_after_n_chars=800
    )
    
    return [c.text for c in chunks if str(c).strip()]


def load_document(path: str) -> list[str]:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return load_and_chunk_pdf(path)
    
    # Fallback to basic text chunking
    text = Path(path).read_text(encoding="utf-8")
    chunks = []
    start = 0
    chunk_size = 800
    overlap = 100
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]


# 2. Embedder (Async)
async def embed_texts(texts: list[str]) -> np.ndarray:
    """Async embedding of chunks."""
    logger.info("embedding_batch", size=len(texts))
    response = await client.aio.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
    )
    vectors = [e.values for e in response.embeddings]
    return np.array(vectors, dtype=np.float32)


# 3. Hybrid Store Wrapper
def save_store(embeddings: np.ndarray, chunks: list[str]) -> None:
    os.makedirs(DB_PATH, exist_ok=True)
    
    # A. ChromaDB Dense Store
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    
    try:
        if collection.count() > 0:
            collection.delete(ids=collection.get()["ids"])
    except Exception:
        pass
    
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.upsert(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=chunks
    )
    logger.info("saved_to_chroma", num_chunks=len(chunks))
    
    # B. BM25 Sparse Store
    tokenized_chunks = [chunk.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)
        
    logger.info("saved_to_bm25", num_chunks=len(chunks))


# 4. Pipeline entry-point
async def ingest(file_paths: list[str]) -> None:
    """Full asynchronous ingestion pipeline with hybrid storage."""
    all_chunks: list[str] = []

    for path in file_paths:
        chunks = load_document(path)
        logger.info("document_processed", path=path, chunks_generated=len(chunks))
        all_chunks.extend(chunks)

    if not all_chunks:
        logger.warning("no_chunks_to_embed")
        return

    embeddings = await embed_texts(all_chunks)
    save_store(embeddings, all_chunks)
    logger.info("ingestion_pipeline_complete")
