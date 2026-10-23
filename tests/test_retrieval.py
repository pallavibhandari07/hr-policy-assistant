"""
tests/test_retrieval.py
────────────────────────
Tests for the retrieval layer (backend/retriever.py).

Tests:
  1. DatabaseNotFoundError is raised when ChromaDB directory is missing.
  2. (Integration) retrieve() returns Documents with required metadata keys.
  3. (Integration) retrieve() returns ≥ 1 result for a known HR topic.

Integration tests require GEMINI_API_KEY and a populated vector store.
They are skipped automatically when the key is not present.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.retriever import DatabaseNotFoundError, retrieve
from tests.conftest import requires_api


# ── Unit test — no API required ───────────────────────────────────────────────

class TestDatabaseNotFoundError:
    def test_missing_chroma_raises_error(self, tmp_path):
        """
        When the ChromaDB directory does not exist, retrieve() must raise
        DatabaseNotFoundError with a helpful message.
        """
        non_existent = tmp_path / "does_not_exist"

        with patch("backend.retriever.settings") as mock_settings:
            mock_settings.validate_chroma_for_query.side_effect = FileNotFoundError(
                f"[config] ChromaDB directory not found at '{non_existent}'. "
                "Run `python ingest_runner.py` first."
            )
            with pytest.raises(DatabaseNotFoundError) as exc_info:
                retrieve("How many days of leave?")

        assert "ingest_runner" in str(exc_info.value).lower() or \
               "not found" in str(exc_info.value).lower()


# ── Integration tests — require API key + populated DB ────────────────────────

@requires_api
class TestRetrieveIntegration:
    """
    These tests run against the live ChromaDB populated by ingest_runner.py.
    Skip them in CI by not setting GEMINI_API_KEY.
    """

    def test_returns_list_of_documents(self):
        from langchain_core.documents import Document
        results = retrieve("How many privilege leave days does an employee get?")
        assert isinstance(results, list)
        assert len(results) >= 1
        assert all(isinstance(d, Document) for d in results)

    def test_documents_have_page_metadata(self):
        results = retrieve("What is the probation period?")
        for doc in results:
            assert "page" in doc.metadata, (
                f"Document missing 'page' in metadata: {doc.metadata}"
            )
            assert isinstance(doc.metadata["page"], int)

    def test_documents_have_source_metadata(self):
        results = retrieve("maternity leave policy")
        for doc in results:
            assert "source" in doc.metadata

    def test_irrelevant_query_still_returns_chunks(self):
        """
        Even for off-topic queries the retriever returns top-K results;
        the chain layer is responsible for the 'Not found' decision.
        """
        results = retrieve("recipe for chocolate cake")
        assert isinstance(results, list)
        # May return some results (nearest neighbours); chain must filter them
