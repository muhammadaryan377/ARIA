"""Configuration for the Insight Agent PDF-RAG capability."""

import os
from pathlib import Path

from core.config import DATA_DIR

RAG_ROOT = DATA_DIR / "insight_rag"
RAG_USERS_DIR = RAG_ROOT / "users"
RAG_USERS_DIR.mkdir(parents=True, exist_ok=True)

RAG_DATABASE_URL = (
    os.getenv("ARIA_RAG_DATABASE_URL")
    or os.getenv("RAG_DATABASE_URL")
    or ""
).strip()

EMBEDDING_MODEL = os.getenv("ARIA_RAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
RAG_LLM_MODEL = os.getenv("ARIA_RAG_LLM_MODEL", "openai/gpt-oss-20b")
RAG_COLLECTION_PREFIX = os.getenv("ARIA_RAG_COLLECTION_PREFIX", "aria_insight_pdf")

MAX_PDF_MB = int(os.getenv("ARIA_RAG_MAX_PDF_MB", "50"))
TEXT_CHUNK_SIZE = int(os.getenv("ARIA_RAG_TEXT_CHUNK_SIZE", "1600"))
TEXT_CHUNK_OVERLAP = int(os.getenv("ARIA_RAG_TEXT_CHUNK_OVERLAP", "250"))
TABLE_ROWS_PER_CHUNK = int(os.getenv("ARIA_RAG_TABLE_ROWS_PER_CHUNK", "25"))
RETRIEVAL_FETCH_K = int(os.getenv("ARIA_RAG_RETRIEVAL_FETCH_K", "18"))
RETRIEVAL_TOP_K = int(os.getenv("ARIA_RAG_RETRIEVAL_TOP_K", "8"))
MAX_CONTEXT_CHARS = int(os.getenv("ARIA_RAG_MAX_CONTEXT_CHARS", "24000"))
RECENT_HISTORY_TURNS = int(os.getenv("ARIA_RAG_RECENT_HISTORY_TURNS", "8"))


def user_key(user_id: str | int) -> str:
    """Return a filesystem/collection-safe stable user key."""
    import hashlib

    return hashlib.sha256(str(user_id).encode("utf-8")).hexdigest()[:20]


def user_dir(user_id: str | int) -> Path:
    path = RAG_USERS_DIR / user_key(user_id)
    path.mkdir(parents=True, exist_ok=True)
    (path / "documents").mkdir(parents=True, exist_ok=True)
    (path / "conversations").mkdir(parents=True, exist_ok=True)
    (path / "chunks").mkdir(parents=True, exist_ok=True)
    return path
