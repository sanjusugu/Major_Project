"""
config.py — Gemini client configuration using the google-genai SDK.
"""

import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

# ── Model names ────────────────────────────────────────────────────────────────
LLM_MODEL       = "models/gemini-2.5-flash"
EMBEDDING_MODEL = "models/gemini-embedding-2-preview"  # latest available

# ── Client ─────────────────────────────────────────────────────────────────────
_api_key = os.getenv("GEMINI_API_KEY")
if not _api_key:
    raise EnvironmentError(
        "GEMINI_API_KEY not found. "
        "Set it in your .env file or as an environment variable."
    )

client = genai.Client(api_key=_api_key)
