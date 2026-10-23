"""
backend/api.py
──────────────
FastAPI application for the HR Policy Assistant.

Routes:
  GET  /          → Serves the frontend index.html
  GET  /health    → Liveness check
  POST /ask       → Accepts a question, returns an AnswerResponse

Error handling:
  - 422  : Pydantic validation (empty / too-long question)
  - 503  : Vector store not initialised (run ingest_runner.py first)
  - 502  : Gemini API failure
  - 500  : Unexpected server errors

CORS is configured for localhost development only.
No secrets are ever included in any response body.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from backend.chain import AnswerResponse as ChainResponse
from backend.chain import answer_question
from backend.retriever import DatabaseNotFoundError

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="HR Policy Assistant",
    description="Evidence-grounded Q&A over the Enso Group HR Policy Manual.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# Serve frontend static files
_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if _FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")


# ── Request / Response schemas ────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="The HR policy question to answer.",
    )

    @field_validator("question")
    @classmethod
    def no_blank_question(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be blank or whitespace only.")
        return v.strip()


class SourceCard(BaseModel):
    page: int
    excerpt: str
    source_file: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceCard]
    is_not_found: bool


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_api_response(chain_resp: ChainResponse) -> AskResponse:
    """Convert the chain's dataclass response to a Pydantic API model."""
    return AskResponse(
        answer=chain_resp.answer,
        is_not_found=chain_resp.is_not_found,
        sources=[
            SourceCard(
                page=s.page,
                excerpt=s.excerpt,
                source_file=s.source_file,
            )
            for s in chain_resp.sources
        ],
    )


# ── Global exception handlers ─────────────────────────────────────────────────

@app.exception_handler(DatabaseNotFoundError)
async def db_not_found_handler(request: Request, exc: DatabaseNotFoundError):
    logger.error("DatabaseNotFoundError: %s", exc)
    return JSONResponse(
        status_code=503,
        content={
            "detail": (
                "The HR policy knowledge base has not been built yet. "
                "Please run `python ingest_runner.py` first."
            )
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected server error occurred. Please try again."},
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the frontend single-page application."""
    index = _FRONTEND_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return FileResponse(str(index))


@app.get("/health", tags=["System"])
async def health_check():
    """Liveness probe — always returns 200 if the server is running."""
    return {"status": "ok", "service": "HR Policy Assistant"}


@app.post("/ask", response_model=AskResponse, tags=["Q&A"])
async def ask(body: AskRequest):
    """
    Answer an HR policy question using the Enso Group HR Manual.

    - Returns the answer with source citations (page number + excerpt).
    - Returns `is_not_found: true` and `answer: "Not found"` when the
      document does not contain sufficient evidence.
    - Never fabricates information.
    - Returns 504 if the AI model exceeds the server-side timeout.
    """
    logger.info("POST /ask | question=%r", body.question[:80])

    TIMEOUT = 60  # seconds — configurable here

    loop = asyncio.get_event_loop()
    try:
        chain_resp = await asyncio.wait_for(
            loop.run_in_executor(None, answer_question, body.question),
            timeout=TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.error("POST /ask timed out after %ds for question=%r", TIMEOUT, body.question[:60])
        raise HTTPException(
            status_code=504,
            detail=f"The AI model did not respond within {TIMEOUT}s. "
                   "The free-tier API may be under load — please try again in a moment.",
        )
    except DatabaseNotFoundError:
        raise  # handled by the global exception handler above
    except Exception as exc:
        exc_name = type(exc).__name__
        if "google" in exc_name.lower() or "api" in exc_name.lower():
            logger.error("Gemini API error: %s", exc)
            raise HTTPException(
                status_code=502,
                detail="The AI service is temporarily unavailable. Please try again.",
            )
        raise  # re-raise for the generic handler

    return _to_api_response(chain_resp)
