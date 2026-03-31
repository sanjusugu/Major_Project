# 🔎 RAG Hallucination Detector (Enterprise Edition)

A high-performance, asynchronous Retrieval-Augmented Generation (RAG) system with a multi-layered hallucination detection pipeline. This system ensures that an LLM's generated answers are strictly grounded in user-provided documents using hybrid retrieval methods and an "LLM-as-a-Judge" grading mechanism.

## ✨ Key Features
- **Async Architecture:** Fully non-blocking FastAPI implementation capable of handling concurrent generation and evaluation tasks.
- **Advanced Document Parsing:** Uses `unstructured` to accurately semantically carve PDFs into coherent sections based on visual titles/headers, replacing blind character counts.
- **Hybrid Search (Dense + Sparse):** Synthesizes Google Gemini Embeddings in **ChromaDB** alongside tokenized **BM25** sparse retrieval for both semantic meaning and precise keyword matching.
- **Cross-Encoder Re-Ranking:** Implements `FlashRank` to strictly re-rank the unionized BM25 and Chroma results, radically increasing contextual awareness.
- **Auto-Gating Hallucination Scoring:** A 4-phase system that evaluates Cosine Similarity between the generated answer and retrieved chunk. If the score is too low, it skips LLM evaluation. Otherwise, it invokes Gemini 2.5 Flash strictly as a Judge, resulting in a **Hybrid Supported/Not-Supported Score**.
- **Modern Web Dashboard:** A beautiful Glassmorphism Vanilla JS/HTML frontend instantly accessible out-of-the-box.

---

## 🏗 System Pipeline (Phases 1-4)
1. **Phase 1 (RAG Engine):** Parses PDF -> BM25 + Chroma Store -> Reranks -> LLM Generates Answer.
2. **Phase 2 (Semantic Similarity):** Quickly calculates mathematical Cosine Similarity between the answer and retrieved source facts.
3. **Phase 3 (LLM Judge):** Asks the LLM to inspect its own answer against the context chunks to declare `SUPPORTED`, `PARTIAL`, or `NOT_SUPPORTED`.
4. **Phase 4 (Hybrid Score):** Generates a definitive fractional final confidence score leveraging both similarity matrices and the logical judge.

---

## 🛠 Prerequisites

1. Python 3.10+
2. A valid Google Gemini API Key

---

## 🚀 Quick Setup & Installation

### 1. Clone & Set Up the Virtual Environment
```bash
git clone https://github.com/sanjusugu/Major_Project.git
cd Major_Project/hallucination-detector
python -m venv venv
```

**Activate the Venv:**
- Windows: `.\venv\Scripts\Activate.ps1`
- Mac/Linux: `source venv/bin/activate`

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
> *Note: By installing `unstructured[pdf]`, it may pull specific python layout packages like `pdfminer.six` internally.*

### 3. Environment Configuration
Create a `.env` file in the root `hallucination-detector/` directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

---

## 💻 Running the Application

### Adding Documents
Place any `.pdf` files you want the model to read into the `data/docs/` directory.

### Boot the Application
Run the FastAPI web application using Uvicorn:
```bash
uvicorn api.main:app --reload --port 8000
```
> The server will automatically detect the `.pdf` files and asynchronously ingest them into ChromaDB and the BM25 Index during startup!

### Access the Web UI Dashboard
Once the server boot log says `Application startup complete`, open your browser to:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

You can dynamically prompt questions across your files and immediately view the generated score breakdown.

---

## 🧪 Automated Testing
If you would like to run head-less verifications natively strictly through terminals without browsers, you can utilize the included testing script:
```bash
python test_api.py
```
This script queries the `/health` endpoint alongside testing **Grounded** specific questions and **Ungrounded** hallucinated queries to verify the pipeline catches invalid references perfectly.
