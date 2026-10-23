"""
backend/config.py
─────────────────
Centralised configuration for the HR Policy Assistant.
All settings are read from environment variables (via .env).

Design note: GEMINI_API_KEY validation is deferred to first access
(via a property) so that unit tests that don't call the Gemini API
can import this module without having the key set.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (two levels up from this file)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


def _optional(key: str, default: str) -> str:
    return os.getenv(key, default).strip() or default


@dataclass(frozen=True)
class Settings:
    # ── Stored as-is; validated lazily via .gemini_api_key property ──
    _gemini_api_key: str = field(repr=False)

    # ── Paths ──────────────────────────────────────────────────
    pdf_path: Path
    chroma_dir: Path

    # ── ChromaDB ───────────────────────────────────────────────
    collection_name: str

    # ── RAG tuning ─────────────────────────────────────────────
    chunk_size: int
    chunk_overlap: int
    top_k: int

    # ── Server ─────────────────────────────────────────────────
    host: str
    port: int

    @property
    def gemini_api_key(self) -> str:
        """Return the API key, raising a clear error if it is not set."""
        if not self._gemini_api_key:
            raise ValueError(
                "[config] Required environment variable 'GEMINI_API_KEY' is missing or empty. "
                "Copy .env.example → .env and fill in your values."
            )
        return self._gemini_api_key

    def validate_paths_for_ingestion(self) -> None:
        """Called before ingestion. Verifies PDF exists."""
        if not self.pdf_path.exists():
            raise FileNotFoundError(
                f"[config] HR Policy PDF not found at '{self.pdf_path}'. "
                f"Place the PDF there and retry."
            )

    def validate_chroma_for_query(self) -> None:
        """Called before querying. Verifies vector store exists."""
        if not self.chroma_dir.exists():
            raise FileNotFoundError(
                f"[config] ChromaDB directory not found at '{self.chroma_dir}'. "
                f"Run `python ingest_runner.py` first to build the vector store."
            )


def _load_settings() -> Settings:
    """Build and return the Settings singleton. Never raises at import time."""
    return Settings(
        _gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        pdf_path=_PROJECT_ROOT / _optional("PDF_PATH", "data/hr_policy.pdf"),
        chroma_dir=_PROJECT_ROOT / _optional("CHROMA_DIR", "vectorstore"),
        collection_name=_optional("COLLECTION_NAME", "hr_policy"),
        chunk_size=int(_optional("CHUNK_SIZE", "800")),
        chunk_overlap=int(_optional("CHUNK_OVERLAP", "150")),
        top_k=int(_optional("TOP_K", "5")),
        host=_optional("HOST", "127.0.0.1"),
        port=int(_optional("PORT", "8000")),
    )


# Module-level singleton — imported by all other modules
settings = _load_settings()
