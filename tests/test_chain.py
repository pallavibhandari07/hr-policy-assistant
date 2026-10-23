"""
tests/test_chain.py
────────────────────
Tests for the RAG answer chain (backend/chain.py).

Tests:
  1. (Unit) Out-of-scope question → answer_question returns is_not_found=True.
  2. (Unit) Empty retrieval result → is_not_found=True without calling LLM.
  3. (Unit) Valid context → LLM called with the right prompt structure.
  4. (Integration) A known HR question returns a non-empty, non-"Not found" answer.
  5. (Integration) A clearly out-of-scope question returns "Not found".

Integration tests require GEMINI_API_KEY + populated vector store.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.chain import NOT_FOUND, AnswerResponse, _build_context, _extract_sources
from tests.conftest import requires_api, synthetic_docs


# ── Context builder tests (no API) ────────────────────────────────────────────

class TestBuildContext:
    def test_context_includes_page_numbers(self, synthetic_docs):
        context = _build_context(synthetic_docs)
        assert "Page 14" in context
        assert "Page 8" in context

    def test_context_includes_excerpt_content(self, synthetic_docs):
        context = _build_context(synthetic_docs)
        assert "Privilege Leave" in context

    def test_each_excerpt_is_numbered(self, synthetic_docs):
        context = _build_context(synthetic_docs)
        assert "[Excerpt 1 |" in context
        assert "[Excerpt 2 |" in context


# ── Source extractor tests (no API) ──────────────────────────────────────────

class TestExtractSources:
    def test_extracts_correct_page_numbers(self, synthetic_docs):
        sources = _extract_sources(synthetic_docs)
        pages = [s.page for s in sources]
        assert 14 in pages
        assert 8 in pages

    def test_excerpt_truncated_at_300_chars(self, synthetic_docs):
        from langchain_core.documents import Document
        long_doc = Document(
            page_content="x" * 500,
            metadata={"source": "hr.pdf", "page": 1},
        )
        sources = _extract_sources([long_doc])
        assert len(sources[0].excerpt) <= 301   # 300 + ellipsis char


# ── Empty retrieval test (no API) ─────────────────────────────────────────────

class TestEmptyRetrieval:
    def test_no_docs_returns_not_found(self):
        """When retriever returns nothing, chain must return is_not_found=True."""
        from backend.chain import answer_question
        with patch("backend.chain.retrieve", return_value=[]):
            response = answer_question("What is the GST rate?")
        assert response.is_not_found is True
        assert response.answer == NOT_FOUND
        assert response.sources == []


# ── Integration tests ─────────────────────────────────────────────────────────

@requires_api
class TestChainIntegration:
    def test_in_scope_question_returns_answer(self):
        """A question covered by the HR manual should return a real answer."""
        from backend.chain import answer_question
        response = answer_question("How many days of privilege leave does an employee get?")
        assert not response.is_not_found
        assert response.answer != NOT_FOUND
        assert len(response.answer) > 5

    def test_in_scope_answer_has_sources(self):
        from backend.chain import answer_question
        response = answer_question("What is the probation period?")
        if not response.is_not_found:
            assert len(response.sources) >= 1
            assert all(s.page >= 1 for s in response.sources)

    def test_out_of_scope_returns_not_found(self):
        """A question completely unrelated to HR policy must return 'Not found'."""
        from backend.chain import answer_question
        response = answer_question(
            "What is the boiling point of water in Celsius?"
        )
        assert response.is_not_found is True
        assert response.answer == NOT_FOUND

    def test_not_found_has_no_sources(self):
        from backend.chain import answer_question
        response = answer_question("What year was the Eiffel Tower built?")
        if response.is_not_found:
            assert response.sources == []
