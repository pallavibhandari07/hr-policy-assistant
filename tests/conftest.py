"""
tests/conftest.py
─────────────────
Shared pytest fixtures for the HR Policy Assistant test suite.

Key design decisions:
  - All tests use a TEMPORARY, isolated ChromaDB directory so they
    never pollute the production vector store.
  - The Gemini API key is required for integration tests; they are
    automatically skipped if GEMINI_API_KEY is not set.
  - A small synthetic document is embedded to avoid relying on the
    real HR PDF (which may not exist in CI).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from langchain_core.documents import Document


# ── Helpers ───────────────────────────────────────────────────────────────────

def has_api_key() -> bool:
    return bool(os.getenv("GEMINI_API_KEY", "").strip())


requires_api = pytest.mark.skipif(
    not has_api_key(),
    reason="GEMINI_API_KEY not set — skipping integration test.",
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def tmp_chroma_dir():
    """Temporary ChromaDB directory, cleaned up after the session."""
    with tempfile.TemporaryDirectory(prefix="hr_test_chroma_") as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="session")
def synthetic_docs() -> list[Document]:
    """
    A minimal set of synthetic HR policy snippets used in unit tests.
    These mirror the structure of real chunks without exposing confidential data.
    """
    return [
        Document(
            page_content=(
                "Section 4.2 — Privilege Leave: Permanent employees are entitled "
                "to 18 days of Privilege Leave (PL) per calendar year."
            ),
            metadata={"source": "hr_policy.pdf", "page": 14, "chunk_id": "synthetic_001"},
        ),
        Document(
            page_content=(
                "Section 3.1 — Probation Period: Employees up to the level of "
                "Senior Manager serve a probation period of 6 months."
            ),
            metadata={"source": "hr_policy.pdf", "page": 8, "chunk_id": "synthetic_002"},
        ),
        Document(
            page_content=(
                "Section 7.4 — Maternity Leave: Female employees are eligible for "
                "26 weeks (182 days) of paid maternity leave as per the Maternity Benefit Act."
            ),
            metadata={"source": "hr_policy.pdf", "page": 31, "chunk_id": "synthetic_003"},
        ),
    ]
