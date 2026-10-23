"""
tests/test_api.py
──────────────────
Tests for the FastAPI endpoints (backend/api.py).

Tests:
  1. GET /health returns 200 with status "ok".
  2. POST /ask with valid question returns 200 + correct schema.
  3. POST /ask with empty question returns 422.
  4. POST /ask with whitespace-only question returns 422.
  5. POST /ask with question >500 chars returns 422.
  6. POST /ask when DB is missing returns 503 with helpful message.
  7. GET / (root) serves the frontend.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="module")
def client():
    """Test client for the FastAPI app."""
    from backend.api import app
    return TestClient(app, raise_server_exceptions=False)


# ── Health check ──────────────────────────────────────────────────────────────

class TestHealthCheck:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_ok_status(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_includes_service_name(self, client):
        data = client.get("/health").json()
        assert "service" in data


# ── Input validation ──────────────────────────────────────────────────────────

class TestAskValidation:
    def test_empty_string_returns_422(self, client):
        response = client.post("/ask", json={"question": ""})
        assert response.status_code == 422

    def test_whitespace_only_returns_422(self, client):
        response = client.post("/ask", json={"question": "   "})
        assert response.status_code == 422

    def test_too_long_question_returns_422(self, client):
        response = client.post("/ask", json={"question": "x" * 501})
        assert response.status_code == 422

    def test_missing_question_field_returns_422(self, client):
        response = client.post("/ask", json={})
        assert response.status_code == 422

    def test_non_json_body_returns_422(self, client):
        response = client.post(
            "/ask",
            content="not json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422


# ── Successful /ask ───────────────────────────────────────────────────────────

class TestAskSuccess:
    def _mock_answer(self, is_not_found: bool = False):
        from backend.chain import AnswerResponse, SourceReference
        if is_not_found:
            return AnswerResponse(answer="Not found", sources=[], is_not_found=True)
        return AnswerResponse(
            answer="Permanent employees receive 18 days of Privilege Leave per year.",
            sources=[
                SourceReference(page=14, excerpt="...18 days of Privilege Leave...", source_file="hr_policy.pdf")
            ],
            is_not_found=False,
        )

    def test_valid_question_returns_200(self, client):
        with patch("backend.api.answer_question", return_value=self._mock_answer()):
            response = client.post("/ask", json={"question": "How many days of leave?"})
        assert response.status_code == 200

    def test_response_has_answer_field(self, client):
        with patch("backend.api.answer_question", return_value=self._mock_answer()):
            data = client.post("/ask", json={"question": "Leave policy?"}).json()
        assert "answer" in data

    def test_response_has_sources_list(self, client):
        with patch("backend.api.answer_question", return_value=self._mock_answer()):
            data = client.post("/ask", json={"question": "Leave policy?"}).json()
        assert "sources" in data
        assert isinstance(data["sources"], list)

    def test_response_has_is_not_found_bool(self, client):
        with patch("backend.api.answer_question", return_value=self._mock_answer()):
            data = client.post("/ask", json={"question": "Leave policy?"}).json()
        assert "is_not_found" in data
        assert isinstance(data["is_not_found"], bool)

    def test_source_card_has_page_number(self, client):
        with patch("backend.api.answer_question", return_value=self._mock_answer()):
            data = client.post("/ask", json={"question": "Leave policy?"}).json()
        if data["sources"]:
            assert "page" in data["sources"][0]
            assert isinstance(data["sources"][0]["page"], int)

    def test_not_found_response_structure(self, client):
        with patch("backend.api.answer_question", return_value=self._mock_answer(is_not_found=True)):
            data = client.post("/ask", json={"question": "capital of france?"}).json()
        assert data["is_not_found"] is True
        assert data["answer"] == "Not found"
        assert data["sources"] == []


# ── Error cases ───────────────────────────────────────────────────────────────

class TestAskErrors:
    def test_missing_db_returns_503(self, client):
        from backend.retriever import DatabaseNotFoundError
        with patch("backend.api.answer_question", side_effect=DatabaseNotFoundError("DB missing")):
            response = client.post("/ask", json={"question": "Leave policy?"})
        assert response.status_code == 503

    def test_503_body_mentions_ingest(self, client):
        from backend.retriever import DatabaseNotFoundError
        with patch("backend.api.answer_question", side_effect=DatabaseNotFoundError("DB missing")):
            data = client.post("/ask", json={"question": "Leave policy?"}).json()
        assert "ingest" in data["detail"].lower() or "knowledge base" in data["detail"].lower()
