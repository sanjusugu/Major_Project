"""
config.py — Gemini client configuration and Observability setup.
"""

import os
from google import genai
from dotenv import load_dotenv
import structlog

load_dotenv()

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
EMBEDDING_MODEL = "models/gemini-embedding-2-preview"

# ── Client ─────────────────────────────────────────────────────────────────────
_api_key = os.getenv("GEMINI_API_KEY")
if not _api_key:
    raise EnvironmentError(
        "GEMINI_API_KEY not found. "
        "Set it in your .env file or as an environment variable."
    )

client = genai.Client(api_key=_api_key)
logger.info("gemini_client_initialized", model=LLM_MODEL, embedding=EMBEDDING_MODEL)
