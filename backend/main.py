"""
backend/main.py
───────────────
FastAPI application factory.
Run with:  uvicorn backend.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from backend.api.chat import router as chat_router
from backend.api.documents import router as documents_router
from backend.api.health import router as health_router
from backend.core.config import get_settings
from backend.db.vector_store import get_vector_store


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Warm up heavy singletons before accepting requests."""
    logger.info("KnowledgeHub AI – starting up …")
    cfg = get_settings()
    # Pre-load the embedding model and ChromaDB client
    vs = get_vector_store()
    logger.info(f"Vector store ready ({vs.total_chunks} chunks in DB)")
    logger.info(f"LLM provider: {cfg.llm_provider} / model: {cfg.llm_model}")
    yield
    logger.info("KnowledgeHub AI – shutting down.")


def create_app() -> FastAPI:
    cfg = get_settings()

    app = FastAPI(
        title="KnowledgeHub AI",
        description=(
            "Multi-document RAG assistant – upload PDFs and ask questions "
            "grounded in your documents."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # ── CORS (permissive for local dev; tighten in production) ────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(health_router)
    app.include_router(documents_router)
    app.include_router(chat_router)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    cfg = get_settings()
    uvicorn.run(
        "backend.main:app",
        host=cfg.api_host,
        port=cfg.api_port,
        reload=True,
        log_level="info",
    )
