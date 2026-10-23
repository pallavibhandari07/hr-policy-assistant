"""
backend/ingest.py
─────────────────
PDF ingestion pipeline for the HR Policy Assistant.

Responsibilities:
  - Load PDF using PyMuPDF, preserving page-number metadata.
  - Split text into overlapping chunks (configurable via settings).
  - Generate deterministic chunk IDs to prevent duplicates on re-runs.
  - Embed chunks in small batches to respect the Gemini free-tier
    rate limit (100 embedContent requests/minute).
  - Write to a persistent ChromaDB collection.

Returns a result dict with page/chunk counts for CLI reporting.
"""

from __future__ import annotations

import hashlib
import time
import logging
from typing import Any

try:
    # Prefer the standalone package when available (avoids deprecation warning)
    from langchain_pymupdf import PyMuPDFLoader  # type: ignore[import]
except ImportError:
    from langchain_community.document_loaders import PyMuPDFLoader  # type: ignore[import]
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

from backend.config import settings

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_chunk_id(source: str, page: int, content_snippet: str) -> str:
    """
    Generate a stable, deterministic ID for a chunk.
    Using SHA-256 of (source path + page number + first 120 chars of content)
    ensures the same chunk always gets the same ID across re-runs.
    """
    raw = f"{source}|page={page}|{content_snippet[:120]}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _load_documents() -> list[Document]:
    """Load the PDF and return one Document per page."""
    loader = PyMuPDFLoader(str(settings.pdf_path))
    docs = loader.load()
    logger.info("Loaded %d pages from %s", len(docs), settings.pdf_path)
    return docs


def _split_documents(docs: list[Document]) -> list[Document]:
    """
    Split page documents into overlapping text chunks.
    Chunk size and overlap are read from settings (configurable via .env).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    logger.info(
        "Split %d pages → %d chunks (size=%d, overlap=%d)",
        len(docs),
        len(chunks),
        settings.chunk_size,
        settings.chunk_overlap,
    )
    return chunks


def _assign_ids(chunks: list[Document]) -> tuple[list[Document], list[str]]:
    """
    Assign a deterministic ID to each chunk.
    Returns the (enriched chunks, list of IDs) in the same order.
    """
    ids: list[str] = []
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        # PyMuPDFLoader stores 0-indexed page numbers; convert to 1-indexed for display
        page = int(chunk.metadata.get("page", 0)) + 1
        chunk.metadata["page"] = page          # store 1-indexed
        chunk_id = _make_chunk_id(source, page, chunk.page_content)
        chunk.metadata["chunk_id"] = chunk_id
        ids.append(chunk_id)
    return chunks, ids


def _get_vectorstore() -> Chroma:
    """Return (or create) the persistent ChromaDB collection."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=settings.gemini_api_key,
    )
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_dir),
    )


# ── Public API ────────────────────────────────────────────────────────────────

def ingest_pdf() -> dict[str, Any]:
    """
    Run the full ingestion pipeline.

    Returns
    -------
    dict with keys:
        pages          : number of PDF pages processed
        chunks_total   : total chunks generated
        chunks_added   : new chunks written to DB
        chunks_skipped : duplicates that were skipped
    """
    # Step 1: Load
    docs = _load_documents()
    num_pages = len(docs)

    # Step 2: Split
    chunks = _split_documents(docs)
    chunks, ids = _assign_ids(chunks)
    num_chunks = len(chunks)

    # Step 3: Connect to vector store
    vectorstore = _get_vectorstore()

    # Step 4: Identify existing IDs to avoid duplicates
    existing_ids: set[str] = set()
    try:
        existing = vectorstore.get(include=[])          # fetch IDs only
        existing_ids = set(existing.get("ids", []))
    except Exception:
        # Collection may be brand-new; that's fine
        pass

    # Step 5: Filter to only new chunks
    new_chunks: list[Document] = []
    new_ids: list[str] = []
    for chunk, cid in zip(chunks, ids):
        if cid not in existing_ids:
            new_chunks.append(chunk)
            new_ids.append(cid)

    chunks_skipped = num_chunks - len(new_chunks)

    # Step 6: Add new chunks in batches (free tier: 100 embed requests/min)
    # Each add_documents call embeds one batch; we sleep between batches.
    BATCH_SIZE = 50          # well under the 100 req/min limit
    SLEEP_SECS = 65          # wait > 60 s between batches to reset the window

    if new_chunks:
        total_batches = (len(new_chunks) + BATCH_SIZE - 1) // BATCH_SIZE
        for batch_num, start in enumerate(range(0, len(new_chunks), BATCH_SIZE), 1):
            batch_docs = new_chunks[start : start + BATCH_SIZE]
            batch_ids  = new_ids[start : start + BATCH_SIZE]
            logger.info(
                "Embedding batch %d/%d (%d chunks)…",
                batch_num, total_batches, len(batch_docs),
            )
            print(
                f"  ⏳  Batch {batch_num}/{total_batches} — "
                f"embedding {len(batch_docs)} chunks…",
                flush=True,
            )
            vectorstore.add_documents(documents=batch_docs, ids=batch_ids)
            # Sleep between batches (not after the last one)
            if start + BATCH_SIZE < len(new_chunks):
                print(
                    f"  ⏸️   Rate-limit pause {SLEEP_SECS}s before next batch…",
                    flush=True,
                )
                time.sleep(SLEEP_SECS)
        logger.info("Added %d new chunks to ChromaDB.", len(new_chunks))
    else:
        logger.info("No new chunks to add — vector store is up to date.")

    return {
        "pages": num_pages,
        "chunks_total": num_chunks,
        "chunks_added": len(new_chunks),
        "chunks_skipped": chunks_skipped,
    }
