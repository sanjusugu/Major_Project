"""
main.py -- End-to-end async pipeline test (Phase 1 -> 4).
Run: python main.py
"""

import os
import asyncio
from pathlib import Path
from rag.ingestion import ingest, DB_PATH
from rag.retriever import Retriever
from rag.generator import generate_answer
from verifier.similarity import compute_similarity
from verifier.scoring import compute_hybrid_score


async def test_run():
    # Phase 1: Ingest
    if not os.path.exists(DB_PATH):
        docs_dir = Path("data/docs")
        files    = [str(p) for p in docs_dir.glob("*") if p.suffix == ".pdf"]
        if not files:
            print("[!] No documents found in data/docs/.")
            return
        print("[*] Ingesting documents asynchronously...\n")
        await ingest(files)
    else:
        print("[OK] ChromaDB already exists -- skipping ingestion.\n")

    # Phase 1: Retrieve + Generate
    question = "What is RAG and how does it reduce hallucinations?"
    print(f"[?] Question: {question}\n")

    retriever      = Retriever()
    hits           = await retriever.retrieve(question, top_k=4)
    context_chunks = [h["chunk"] for h in hits]

    print(f"[*] Retrieved {len(hits)} chunks:")
    for i, h in enumerate(hits, 1):
        print(f"  [{i}] score={h['score']:.4f}  -> {h['chunk'][:90].strip()} ...")

    print("\n[*] Generating answer with Gemini 2.5 Flash...")
    ans_res = await generate_answer(question, context_chunks)
    answer = ans_res["answer"]
    print(f"\n{'='*60}\nANSWER:\n{'='*60}\n{answer}")

    # Phase 2: Similarity
    print("\n[*] Computing similarity (Phase 2)...")
    sim = await compute_similarity(answer, context_chunks)
    print(f"\n{'='*60}\nSIMILARITY (Phase 2):\n{'='*60}")
    print(f"  Max Score  : {sim['max_score']}  -> {sim['label']}")
    print(f"  Mean Score : {sim['mean_score']}")
    print(f"  Per-chunk  : {sim['chunk_scores']}")

    # Phase 3 + 4: Hybrid scoring
    print("\n[*] Computing hybrid score (Phase 3 + 4)...")
    hybrid = await compute_hybrid_score(answer, context_chunks, sim_result=sim)
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
    asyncio.run(test_run())
