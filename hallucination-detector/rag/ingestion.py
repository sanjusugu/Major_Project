"""
rag/ingestion.py -- Document ingestion pipeline.

Steps:
  1. Load text from .txt or .pdf files
  2. Chunk into overlapping segments
  3. Embed each chunk via Gemini embedding model
  4. Store in a FAISS index (saved to disk)
"""

import os
import pickle
import numpy as np
import faiss
from pathlib import Path
from pypdf import PdfReader
from config import client, EMBEDDING_MODEL

# Paths
INDEX_PATH  = "data/faiss.index"
CHUNKS_PATH = "data/chunks.pkl"


# 1. Loaders
def load_txt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_document(path: str) -> str:
    """Load a .txt or .pdf file and return raw text."""
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return load_pdf(path)
    return load_txt(path)


# 2. Chunker
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """
    Split text into overlapping chunks by character count.
    chunk_size=800 chars ~150-200 tokens; overlap preserves context boundaries.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]  # drop empty chunks


# 3. Embedder
def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a list of texts using the Gemini embedding model.
    Returns a float32 numpy array of shape (N, D).
    """
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
    )
    vectors = [e.values for e in response.embeddings]
    return np.array(vectors, dtype=np.float32)


# 4. FAISS store
def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build an inner-product (cosine after normalisation) FAISS index."""
    faiss.normalize_L2(embeddings)          # normalise for cosine similarity
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_store(index: faiss.IndexFlatIP, chunks: list[str]) -> None:
    os.makedirs("data", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print(f"[OK] Saved {len(chunks)} chunks -> {INDEX_PATH}")


def load_store() -> tuple[faiss.IndexFlatIP, list[str]]:
    index  = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


# 5. Pipeline entry-point
def ingest(file_paths: list[str]) -> None:
    """
    Full ingestion pipeline:
      load -> chunk -> embed -> build FAISS index -> save to disk
    """
    all_chunks: list[str] = []

    for path in file_paths:
        print(f"[*] Loading: {path}")
        text   = load_document(path)
        chunks = chunk_text(text)
        print(f"    -> {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"\n[*] Embedding {len(all_chunks)} chunks ...")
    embeddings = embed_texts(all_chunks)

    print("[*] Building FAISS index ...")
    index = build_index(embeddings)

    save_store(index, all_chunks)


if __name__ == "__main__":
    # Quick test: ingest every file in data/docs/
    docs_dir = Path("data/docs")
    docs_dir.mkdir(parents=True, exist_ok=True)
    files = [str(p) for p in docs_dir.glob("*") if p.suffix in {".txt", ".pdf"}]
    if not files:
        print("[!] No documents found in data/docs/. Add .txt or .pdf files first.")
    else:
        ingest(files)
