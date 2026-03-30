"""
main.py -- End-to-end pipeline test (Phase 1 -> 4).
Run: python main.py
"""

import os
from pathlib import Path
from rag.ingestion import ingest
from rag.retriever import Retriever
from rag.generator import generate_answer
from verifier.similarity import compute_similarity
from verifier.scoring import compute_hybrid_score

INDEX_PATH = "data/faiss.index"


def main():
    # Phase 1: Ingest
    if not os.path.exists(INDEX_PATH):
        docs_dir = Path("data/docs")
        files    = [str(p) for p in docs_dir.glob("*") if p.suffix in {".txt", ".pdf"}]
        if not files:
            print("[!] No documents found in data/docs/.")
            return
        print("[*] Ingesting documents...\n")
        ingest(files)
    else:
        print("[OK] FAISS index already exists -- skipping ingestion.\n")

    # Phase 1: Retrieve + Generate
    question = "What is RAG and how does it reduce hallucinations?"
    print(f"[?] Question: {question}\n")

    retriever      = Retriever()
    hits           = retriever.retrieve(question, top_k=4)
    context_chunks = [h["chunk"] for h in hits]

    print(f"[*] Retrieved {len(hits)} chunks:")
    for i, h in enumerate(hits, 1):
        print(f"  [{i}] score={h['score']:.4f}  -> {h['chunk'][:90].strip()} ...")

    print("\n[*] Generating answer with Gemini 2.5 Flash...")
    answer = generate_answer(question, context_chunks)["answer"]
    print(f"\n{'='*60}\nANSWER:\n{'='*60}\n{answer}")

    # Phase 2: Similarity
    print("\n[*] Computing similarity (Phase 2)...")
    sim = compute_similarity(answer, context_chunks)
    print(f"\n{'='*60}\nSIMILARITY (Phase 2):\n{'='*60}")
    print(f"  Max Score  : {sim['max_score']}  -> {sim['label']}")
    print(f"  Mean Score : {sim['mean_score']}")
    print(f"  Per-chunk  : {sim['chunk_scores']}")

    # Phase 3 + 4: Hybrid scoring
    print("\n[*] Computing hybrid score (Phase 3 + 4)...")
    hybrid = compute_hybrid_score(answer, context_chunks, sim_result=sim)
    judge  = hybrid["judge"]

    print(f"\n{'='*60}\nHYBRID SCORE (Phase 4):\n{'='*60}")
    print(f"  Gating Label  : {hybrid['gating_label']}")
    print(f"  Judge was run : {hybrid['judge_was_run']}")
    if judge:
        print(f"  Verdict       : {judge['verdict']}  (conf={judge['confidence']})")
        print(f"  Judge Score   : {judge['judge_score']}")
        print(f"  Explanation   : {judge['explanation']}")
    print(f"  FINAL SCORE   : {hybrid['final_score']}")
    print("="*60)


if __name__ == "__main__":
    main()
