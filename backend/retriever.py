"""
backend/retriever.py
─────────────────────
Retrieval wrapper around the persistent ChromaDB vector store.

Responsibilities:
  - Connect to the existing ChromaDB collection built by ingest.py.
  - Accept a natural-language query and return the top-K most
    relevant document chunks with their metadata.
  - Raise a clear, actionable error if the vector store is missing
    (i.e., ingestion has not been run yet).
"""

from __future__ import annotations

import logging

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from backend.config import settings

logger = logging.getLogger(__name__)


class DatabaseNotFoundError(RuntimeError):
    """Raised when the ChromaDB vector store has not been initialised."""


def _get_vectorstore() -> Chroma:
    """
    Connect to the persistent ChromaDB collection.
    Raises DatabaseNotFoundError if the directory does not exist.
    """
    try:
        settings.validate_chroma_for_query()
    except FileNotFoundError as exc:
        raise DatabaseNotFoundError(str(exc)) from exc

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=settings.gemini_api_key,
    )
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_dir),
    )


def retrieve(query: str) -> list[Document]:
    """
    Perform similarity search against the HR policy vector store.

    Parameters
    ----------
    query : str
        The user's natural-language question.

    Returns
    -------
    list[Document]
        Top-K document chunks sorted by relevance, each with metadata:
          - page       : 1-indexed page number in the PDF
          - source     : path to the source PDF
          - chunk_id   : deterministic chunk identifier
          - (any other metadata preserved by PyMuPDFLoader)

    Raises
    ------
    DatabaseNotFoundError
        If the vector store directory does not exist.
    """
    vectorstore = _get_vectorstore()
    results = vectorstore.similarity_search(query, k=settings.top_k)
    logger.info("Retrieved %d chunks for query: %r", len(results), query[:80])
    return results
