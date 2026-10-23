"""
tests/test_ingest.py
─────────────────────
Tests for the PDF ingestion pipeline (backend/ingest.py).

Tests:
  1. _make_chunk_id produces a deterministic, stable hash.
  2. _make_chunk_id produces different IDs for different inputs.
  3. _assign_ids adds chunk_id and 1-indexed page to metadata.
  4. _split_documents honours configured chunk size.
  5. Re-running ingest (simulated) does NOT increase chunk count (idempotency).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.ingest import _assign_ids, _make_chunk_id, _split_documents


# ── chunk ID tests ────────────────────────────────────────────────────────────

class TestMakeChunkId:
    def test_deterministic(self):
        """Same inputs → same ID every time."""
        id1 = _make_chunk_id("hr_policy.pdf", 5, "Some policy text here.")
        id2 = _make_chunk_id("hr_policy.pdf", 5, "Some policy text here.")
        assert id1 == id2

    def test_different_page_different_id(self):
        id1 = _make_chunk_id("hr_policy.pdf", 5, "Same text.")
        id2 = _make_chunk_id("hr_policy.pdf", 6, "Same text.")
        assert id1 != id2

    def test_different_content_different_id(self):
        id1 = _make_chunk_id("hr_policy.pdf", 1, "Text A")
        id2 = _make_chunk_id("hr_policy.pdf", 1, "Text B")
        assert id1 != id2

    def test_id_is_32_chars(self):
        cid = _make_chunk_id("hr.pdf", 1, "content")
        assert len(cid) == 32
        assert cid.isalnum()


# ── metadata enrichment tests ─────────────────────────────────────────────────

class TestAssignIds:
    def _make_doc(self, page: int, content: str) -> Document:
        return Document(
            page_content=content,
            metadata={"source": "hr_policy.pdf", "page": page},
        )

    def test_chunk_id_added_to_metadata(self):
        docs = [self._make_doc(0, "Leave policy content.")]
        enriched, ids = _assign_ids(docs)
        assert "chunk_id" in enriched[0].metadata
        assert enriched[0].metadata["chunk_id"] == ids[0]

    def test_page_converted_to_1_indexed(self):
        """PyMuPDFLoader gives 0-indexed pages; _assign_ids must convert to 1-indexed."""
        docs = [self._make_doc(0, "Page zero content.")]
        enriched, _ = _assign_ids(docs)
        assert enriched[0].metadata["page"] == 1

    def test_ids_match_chunk_count(self):
        docs = [self._make_doc(i, f"Content {i}") for i in range(5)]
        enriched, ids = _assign_ids(docs)
        assert len(ids) == 5
        assert len(enriched) == 5


# ── text splitting tests ──────────────────────────────────────────────────────

class TestSplitDocuments:
    def _make_long_doc(self, chars: int) -> Document:
        content = ("HR policy text. " * 100)[:chars]
        return Document(
            page_content=content,
            metadata={"source": "hr_policy.pdf", "page": 1},
        )

    def test_long_document_produces_multiple_chunks(self):
        """A document longer than chunk_size should produce >1 chunks."""
        doc = self._make_long_doc(3000)   # well above default 800
        chunks = _split_documents([doc])
        assert len(chunks) > 1

    def test_short_document_stays_as_one_chunk(self):
        """A document shorter than chunk_size stays as a single chunk."""
        doc = self._make_long_doc(100)
        chunks = _split_documents([doc])
        assert len(chunks) == 1

    def test_metadata_preserved_across_chunks(self):
        doc = self._make_long_doc(3000)
        chunks = _split_documents([doc])
        for chunk in chunks:
            assert chunk.metadata.get("source") == "hr_policy.pdf"


# ── idempotency simulation ────────────────────────────────────────────────────

class TestIdempotency:
    def _make_docs(self):
        """Create a fresh set of documents (mimics reloading from PDF)."""
        return [
            Document(
                page_content="Leave entitlement policy.",
                metadata={"source": "hr.pdf", "page": 0},
            )
        ]

    def test_no_duplicate_ids(self):
        """
        Running _assign_ids on equivalent documents (same source/page/content)
        must produce identical IDs — ensuring re-ingestion skips duplicates.
        """
        _, ids_run1 = _assign_ids(self._make_docs())
        _, ids_run2 = _assign_ids(self._make_docs())
        assert ids_run1 == ids_run2, (
            "Same document content must always produce the same chunk ID."
        )

    def test_no_duplicate_ids_within_run(self):
        """Each unique chunk gets a unique ID within a single ingestion run."""
        docs = [
            Document(page_content=f"Policy section {i}.", metadata={"source": "hr.pdf", "page": i})
            for i in range(5)
        ]
        _, ids = _assign_ids(docs)
        assert len(set(ids)) == len(ids), "All chunk IDs in a run must be unique."
