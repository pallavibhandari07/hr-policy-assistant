"""
run.py
──────
CLI entry point for the FastAPI server.

Usage:
    python run.py

Reads HOST and PORT from .env (defaults: 127.0.0.1:8000).
Open http://127.0.0.1:8000 in your browser after starting.
"""

import uvicorn

from backend.config import settings

if __name__ == "__main__":
    print("=" * 60)
    print("  HR Policy Assistant — API Server")
    print("=" * 60)
    print(f"\n🚀  Starting on http://{settings.host}:{settings.port}")
    print("    Press Ctrl+C to stop.\n")

    uvicorn.run(
        "backend.api:app",
        host=settings.host,
        port=settings.port,
        reload=True,          # auto-reload on code changes during development
        log_level="info",
    )
