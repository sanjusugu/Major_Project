import requests
import json
import time

API_URL = "http://127.0.0.1:8000"

def test_health():
    print("=== Testing /health ===")
    try:
        response = requests.get(f"{API_URL}/health")
        response.raise_for_status()
        print(json.dumps(response.json(), indent=2))
        return True
    except requests.exceptions.RequestException as e:
        print(f"Health check failed: {e}")
        return False

def test_ask(question: str):
    print(f"\n=== Testing /ask ===")
    print(f"Question: '{question}'\n")
    
    payload = {
        "question": question,
        "top_k": 4
    }
    
    start_time = time.time()
    try:
        response = requests.post(f"{API_URL}/ask", json=payload)
        response.raise_for_status()
        data = response.json()
        
        print(f"Elapsed time: {time.time() - start_time:.2f} seconds\n")
        
        print("--- ANSWER ---")
        print(data.get("answer", "No answer found"))
        print("\n--- SIMILARITY (Phase 2) ---")
        sim = data.get("similarity", {})
        print(f"Label: {sim.get('label')}")
        print(f"Max Score: {sim.get('max_score')}")
        
        print("\n--- LLM JUDGE (Phase 3) ---")
        judge = data.get("judge")
        if judge:
            print(f"Verdict: {judge.get('verdict')}")
            print(f"Confidence: {judge.get('confidence')}")
            print(f"Explanation: {judge.get('explanation')}")
        else:
            print("Judge skipped (Gated).")
            
        print("\n--- HYBRID SCORE (Phase 4) ---")
        hybrid = data.get("hybrid", {})
        print(f"Final Score: {hybrid.get('final_score')}")
        print(f"Gating Label: {hybrid.get('gating_label')}")
        print(f"Judge Was Run: {hybrid.get('judge_was_run')}")
        
    except requests.exceptions.RequestException as e:
        print(f"Ask check failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Error details: {e.response.text}")

if __name__ == "__main__":
    if test_health():
        # First question is grounded in the sample PDF
        test_ask("What is RAG and how does it reduce hallucination?")
        
        # Second question may not be in the PDF
        test_ask("What are the system requirements for installing Python?")
