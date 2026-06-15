"""
backend/api/chat.py
───────────────────
FastAPI router for the RAG question-answering endpoint.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from loguru import logger

from backend.core.llm_client import get_llm_client
from backend.db.vector_store import get_vector_store
from backend.models.schemas import ChatRequest, ChatResponse, SourceChunk

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    RAG pipeline:
      1. Retrieve top-k relevant chunks from ChromaDB.
      2. Build a prompt with context and chat history.
      3. Call the LLM to generate an answer.
      4. Return the answer with source citations.
    """
    cfg_top_k = request.top_k

    # ── Retrieval ─────────────────────────────────────────────────────────────
    try:
        raw_results = get_vector_store().query(
            question=request.question,
            top_k=cfg_top_k,
            doc_ids=request.doc_ids if request.doc_ids else None,
        )
    except Exception as exc:
        logger.exception(f"Vector search failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Vector search failed. Please try again.",
        )

    if not raw_results:
        # No chunks found – answer without context
        answer = (
            "I could not find relevant information in the uploaded documents "
            "to answer your question. Please upload relevant PDFs first."
        )
        return ChatResponse(
            answer=answer,
            sources=[],
            model_used="n/a",
            tokens_used=None,
        )

    # ── LLM Generation ────────────────────────────────────────────────────────
    try:
        llm = get_llm_client()
        answer_text, tokens_used = llm.answer(
            question=request.question,
            context_chunks=raw_results,
            history=request.history,
        )
    except Exception as exc:
        logger.exception(f"LLM call failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM generation failed: {str(exc)}",
        )

    # ── Build source citations ────────────────────────────────────────────────
    sources = [
        SourceChunk(
            doc_id=r["doc_id"],
            filename=r["filename"],
            page_number=r["page_number"],
            chunk_index=r["chunk_index"],
            excerpt=r["text"][:300] + ("…" if len(r["text"]) > 300 else ""),
            relevance_score=r["score"],
        )
        for r in raw_results
    ]

    logger.info(
        f"Chat answered | question_len={len(request.question)} "
        f"sources={len(sources)} tokens={tokens_used}"
    )

    return ChatResponse(
        answer=answer_text,
        sources=sources,
        model_used=get_llm_client()._model,
        tokens_used=tokens_used,
    )
