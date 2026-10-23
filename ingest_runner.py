"""
ingest_runner.py
────────────────
CLI entry point for the ingestion pipeline.

Usage:
    python ingest_runner.py

What it does:
  1. Validates that the HR Policy PDF exists.
  2. Loads and chunks the PDF with page-level metadata.
  3. Writes chunks to the persistent ChromaDB vector store.
  4. Is idempotent — re-running will NOT duplicate chunks.
"""

import sys
import time

from backend.config import settings
from backend.ingest import ingest_pdf


def main() -> None:
    print("=" * 60)
    print("  HR Policy Assistant — Ingestion Pipeline")
    print("=" * 60)

    # Validate PDF exists before doing anything
    try:
        settings.validate_paths_for_ingestion()
    except FileNotFoundError as exc:
        print(f"\n❌  {exc}")
        sys.exit(1)

    print(f"\n📄  PDF      : {settings.pdf_path}")
    print(f"🗄️   Vector DB: {settings.chroma_dir}")
    print(f"📦  Chunk size / overlap: {settings.chunk_size} / {settings.chunk_overlap} chars")
    print(f"🔗  Collection: {settings.collection_name}\n")

    start = time.perf_counter()

    try:
        result = ingest_pdf()
    except Exception as exc:
        print(f"\n❌  Ingestion failed: {exc}")
        sys.exit(1)

    elapsed = time.perf_counter() - start

    print(f"\n✅  Ingestion complete in {elapsed:.1f}s")
    print(f"    Pages processed : {result['pages']}")
    print(f"    Chunks created  : {result['chunks_total']}")
    print(f"    New chunks added: {result['chunks_added']}")
    print(f"    Skipped (dupes) : {result['chunks_skipped']}")
    print("\nYou can now start the API server with:  python run.py\n")


if __name__ == "__main__":
    main()
