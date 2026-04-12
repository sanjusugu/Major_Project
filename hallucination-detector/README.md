# 🔎 RAG Hallucination Detector (Industrial Enterprise Edition)

A professional, high-performance, asynchronous Retrieval-Augmented Generation (RAG) system with a multi-layered hallucination detection pipeline. This system ensures that an LLM's generated answers are strictly grounded in user-provided documents using hybrid retrieval methods, local intelligence, and a logical "LLM-as-a-Judge" grading mechanism.

## ✨ Industrial Features (New)
- **MongoDB Persistence:** Switched from local files to a scalable **MongoDB** data layer for document chunks and vector storage.
- **Local Embedded Intelligence:** Uses the **Sentence-Transformers (`all-MiniLM-L6-v2`)** model running locally. No more cloud latency or API costs for embedding generation.
- **Async Pipeline:** Fully non-blocking FastAPI implementation capable of handling concurrent generation and evaluation tasks.
- **Advanced Document Parsing:** Uses `unstructured` to semantically chunk PDFs by actual titles and headers.
- **Hybrid Search (Dense + Sparse):** Combines **MongoDB Vector Search** with tokenized **BM25** for high-precision retrieval.
- **Cross-Encoder Re-Ranking:** Implements `FlashRank` to strictly refine retrieval results before they reach the LLM.
- **Auto-Gating Hallucination Scoring:** A 4-phase system that evaluates Cosine Similarity and invokes Gemini 2.5 Flash strictly as a Judge for a final **Hybrid Safety Score**.
- **Modern Dashboard:** A beautiful Glassmorphism UI accessible out-of-the-box.

---

## 🏗 System Pipeline (Phases 1-4)
1. **Phase 1 (RAG Engine):** Parses PDF -> BM25 + MongoDB Store -> Reranks -> LLM Generates Answer.
2. **Phase 2 (Semantic Similarity):** Calculates mathematical Cosine Similarity using **local** embedding models.
3. **Phase 3 (LLM Judge):** Gemini 2.5 Flash inspects the logical grounding of claims against Source Context.
4. **Phase 4 (Hybrid Score):** Synthesizes logical analysis and mathematical similarity into a final safety metric.

---

## 🛠 Prerequisites
1. Python 3.10+
2. **MongoDB** (Local or Atlas instance)
3. Google Gemini API Key

---

## 🚀 Quick Setup & Installation

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/sanjusugu/Major_Project.git
cd Major_Project/hallucination-detector
python -m venv venv
```

**Activate Venv:**
- Windows: `.\venv\Scripts\Activate.ps1`
- Mac/Linux: `source venv/bin/activate`

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
MONGO_URI=mongodb://localhost:27017/  # Optional: Defaults to local
```

---

## 💻 Running the Application

### Adding Documents
Place any `.pdf` files into the `data/docs/` directory.

### Launch the System
Launch the integrated server and database connection:
```bash
python main.py
```
> On the first run, the local embedding model will download, and your PDFs will be automatically ingested into MongoDB.

### Access the Web UI Dashboard
Open your browser to:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 🧪 Automated Testing
Run the headless verification script to check Grounded vs. Hallucinated responses:
```bash
python test_api.py
```
