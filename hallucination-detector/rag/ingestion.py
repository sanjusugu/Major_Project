"""
rag/ingestion.py -- Smart incremental ingestion pipeline using MongoDB + BM25 + Unstructured.

On each run:
  - Checks which PDF files are already indexed in MongoDB (by source filename).
  - Only processes NEW files that are not yet indexed.
  - Rebuilds the BM25 sparse index from ALL chunks currently in MongoDB.
"""

import os
import numpy as np
import pickle
from pathlib import Path

# Advanced Parser Imports
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

from rank_bm25 import BM25Okapi
from config import embed_model, logger, collection

# Paths
BM25_PATH = "data/bm25.pkl"


# ── 1. Loaders / Chunkers ─────────────────────────────────────────────────────

def load_and_chunk_pdf(path: str) -> list[str]:
    """Semantic layout-aware chunking using unstructured."""
    logger.info("partitioning_pdf", path=path)
    elements = partition_pdf(path, strategy="fast")
    chunks   = chunk_by_title(
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
    # Fallback plain-text chunking
    text   = Path(path).read_text(encoding="utf-8")
    chunks, start = [], 0
    while start < len(text):
        end = min(start + 800, len(text))
        chunks.append(text[start:end].strip())
        start += 700
    return [c for c in chunks if c]


# ── 2. Embedder ───────────────────────────────────────────────────────────────

async def embed_texts(texts: list[str]) -> np.ndarray:
    """Generate embeddings locally via the pre-loaded Sentence-Transformer model."""
    logger.info("embedding_batch_local", size=len(texts))
    return np.array(embed_model.encode(texts), dtype=np.float32)


# ── 3. Smart Incremental Ingestion ───────────────────────────────────────────

def get_indexed_sources() -> set[str]:
    """Return set of source filenames already stored in MongoDB."""
    sources = collection.distinct("source")
    return set(sources)


def filter_new_files(file_paths: list[str]) -> list[str]:
    """Return only the files that are NOT yet indexed in MongoDB."""
    indexed = get_indexed_sources()
    new_files = []
    for path in file_paths:
        name = Path(path).name
        if name in indexed:
            logger.info("skipping_already_indexed", file=name)
        else:
            new_files.append(path)
    return new_files


def save_new_chunks(chunks: list[str], source: str, start_index: int, embeddings: np.ndarray) -> None:
    """Append new chunks from a single source file into MongoDB."""
    documents = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings.tolist())):
        documents.append({
            "chunk_id": f"{source}__chunk_{start_index + i}",
            "source":   source,
            "text":     chunk,
            "embedding": embedding,
        })
    if documents:
        collection.insert_many(documents)
    logger.info("saved_chunks_to_mongodb", source=source, count=len(documents))


def rebuild_bm25_from_mongodb() -> None:
    """Rebuild the full BM25 sparse index from ALL chunks currently in MongoDB."""
    all_docs = list(collection.find({}, {"text": 1, "_id": 0}))
    all_texts = [d["text"] for d in all_docs]
    if not all_texts:
        logger.warning("no_chunks_in_mongodb_for_bm25")
        return
    tokenized = [t.lower().split() for t in all_texts]
    bm25 = BM25Okapi(tokenized)
    os.makedirs(os.path.dirname(BM25_PATH), exist_ok=True)
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": all_texts}, f)
    logger.info("bm25_rebuilt", total_chunks=len(all_texts))


# ── 4. Pipeline Entry-Point ───────────────────────────────────────────────────

async def ingest(file_paths: list[str], force: bool = False) -> dict:
    """
    Smart incremental ingestion pipeline.
    - If force=True, all files are re-ingested (existing data for those files deleted).
    - If force=False (default), only NEW files not yet indexed are processed.
    Returns: dict with counts of new/skipped files.
    """
    total_new_chunks = 0
    skipped_files    = []
    ingested_files   = []

    for path in file_paths:
        source = Path(path).name

        if not force and source in get_indexed_sources():
            logger.info("skipping_already_indexed", file=source)
            skipped_files.append(source)
            continue

        if force:
            # Remove existing chunks for this file before re-indexing
            collection.delete_many({"source": source})
            logger.info("force_reindex_cleared", file=source)

        chunks = load_document(path)
        logger.info("document_processed", path=path, chunks_generated=len(chunks))

        if not chunks:
            logger.warning("no_chunks_extracted", path=path)
            continue

        # Current total docs in mongo to create unique global chunk_ids
        current_count = collection.count_documents({})

        embeddings = await embed_texts(chunks)
        save_new_chunks(chunks, source, current_count, embeddings)

        total_new_chunks += len(chunks)
        ingested_files.append(source)

    if ingested_files:
        # Rebuild full BM25 index from all MongoDB chunks
        rebuild_bm25_from_mongodb()
        logger.info("ingestion_complete", new_files=len(ingested_files), new_chunks=total_new_chunks)
    else:
        logger.info("no_new_files_to_ingest", skipped=len(skipped_files))

    return {
        "ingested": ingested_files,
        "skipped":  skipped_files,
        "new_chunks": total_new_chunks,
    }
