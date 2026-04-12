"""
config.py — Gemini client configuration and Observability setup.
"""

import os
from google import genai
from dotenv import load_dotenv
import structlog

load_dotenv()

# ── MongoDB Setup ─────────────────────────────────────────────────────────────
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB_NAME = "hallucination_detector"
MONGO_COLLECTION_NAME = "rag_chunks"
MONGO_VECTOR_INDEX_NAME = "vector_index"

# ── Observability Setup ───────────────────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
)
logger = structlog.get_logger()

# ── Model names ────────────────────────────────────────────────────────────────
LLM_MODEL       = "models/gemini-2.5-flash"
LOCAL_EMBED_MODEL = "all-MiniLM-L6-v2"

# ── Client ─────────────────────────────────────────────────────────────────────
_api_key = os.getenv("GEMINI_API_KEY")
if not _api_key:
    raise EnvironmentError(
        "GEMINI_API_KEY not found. "
        "Set it in your .env file or as an environment variable."
    )

from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

try:
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client[MONGO_DB_NAME]
    collection = db[MONGO_COLLECTION_NAME]
    logger.info("mongodb_client_initialized", uri="HIDDEN", db=MONGO_DB_NAME)
except Exception as e:
    logger.error("mongodb_initialization_failed", error=str(e))

# Load local embedding model once
logger.info("loading_local_embedding_model", model=LOCAL_EMBED_MODEL)
embed_model = SentenceTransformer(LOCAL_EMBED_MODEL)
logger.info("local_embedding_model_ready")

client = genai.Client(api_key=_api_key)
logger.info("gemini_client_initialized", model=LLM_MODEL, embedding=LOCAL_EMBED_MODEL)
