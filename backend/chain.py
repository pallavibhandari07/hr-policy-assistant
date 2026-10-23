"""
backend/chain.py
────────────────
RAG answer-generation chain for the HR Policy Assistant.

Responsibilities:
  - Build a LangChain LCEL chain: retriever → prompt → Gemini LLM.
  - Enforce a strict evidence-only prompt: the LLM must answer
    exclusively from retrieved context or respond "Not found".
  - Extract source citations (page number + excerpt) from retrieved chunks.
  - Return a structured AnswerResponse dataclass.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

from backend.config import settings
from backend.retriever import retrieve

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

NOT_FOUND = "Not found"

SYSTEM_PROMPT = """\
You are an HR Policy Assistant for the Enso Group.
Your ONLY source of information is the document excerpts provided below.

Rules you MUST follow without exception:
1. Answer the question using ONLY the information in the excerpts.
2. Do NOT use your general training knowledge to supplement or guess.
3. If the excerpts do not contain enough information to answer the question,
   respond with exactly these two words and nothing else: Not found
4. When you do answer, be concise and factual.
5. Do not add disclaimers, hedges, or phrases like "According to the document".
"""

HUMAN_PROMPT = """\
DOCUMENT EXCERPTS:
{context}

QUESTION:
{question}

ANSWER:"""


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class SourceReference:
    page: int
    excerpt: str
    source_file: str


@dataclass
class AnswerResponse:
    answer: str
    sources: list[SourceReference] = field(default_factory=list)
    is_not_found: bool = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_context(docs: list[Document]) -> str:
    """Format retrieved chunks into a numbered context block."""
    parts = []
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page", "?")
        parts.append(f"[Excerpt {i} | Page {page}]\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(parts)


def _extract_sources(docs: list[Document]) -> list[SourceReference]:
    """Build SourceReference objects from document metadata."""
    sources = []
    for doc in docs:
        page = doc.metadata.get("page", 0)
        source_file = Path(doc.metadata.get("source", "hr_policy.pdf")).name
        excerpt = doc.page_content.strip()
        # Truncate long excerpts for display
        if len(excerpt) > 300:
            excerpt = excerpt[:297] + "…"
        sources.append(SourceReference(page=page, excerpt=excerpt, source_file=source_file))
    return sources


def _build_chain():
    """Construct and return the LCEL RAG chain."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-lite-latest",   # confirmed ~4s on free tier (vs 44-212s for gemini-3.8-flash)
        google_api_key=settings.gemini_api_key,
        temperature=0,          # deterministic, fact-based answers
        max_output_tokens=512,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ])
    return prompt | llm | StrOutputParser()


# ── Public API ────────────────────────────────────────────────────────────────

def answer_question(question: str) -> AnswerResponse:
    """
    Run the full RAG pipeline for a user question.

    Parameters
    ----------
    question : str
        The user's natural-language HR policy question.

    Returns
    -------
    AnswerResponse
        answer        : The LLM's answer, or "Not found".
        sources       : List of source references with page numbers.
        is_not_found  : True when no evidence was found.

    Raises
    ------
    DatabaseNotFoundError
        If the vector store has not been initialised (re-raised from retriever).
    google.api_core.exceptions.GoogleAPIError
        On Gemini API failures.
    """
    # 1. Retrieve relevant chunks
    docs = retrieve(question)
    logger.info("Retrieved %d chunks for question: %r", len(docs), question[:80])

    if not docs:
        logger.warning("No chunks retrieved — returning Not found.")
        return AnswerResponse(answer=NOT_FOUND, sources=[], is_not_found=True)

    # 2. Build context and run the chain
    context = _build_context(docs)
    chain = _build_chain()

    raw_answer: str = chain.invoke({
        "context": context,
        "question": question,
    })

    answer = raw_answer.strip()
    logger.info("LLM answer: %r", answer[:120])

    # 3. Normalise "Not found" variants to the canonical form
    normalised = answer.lower().replace(".", "").strip()
    if normalised == "not found":
        return AnswerResponse(answer=NOT_FOUND, sources=[], is_not_found=True)

    # 4. Extract citations
    sources = _extract_sources(docs)

    return AnswerResponse(answer=answer, sources=sources, is_not_found=False)
