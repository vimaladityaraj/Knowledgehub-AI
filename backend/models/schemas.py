"""
backend/models/schemas.py
─────────────────────────
Pydantic request / response schemas shared across the API.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


# ── Document ──────────────────────────────────────────────────────────────────

class DocumentMetadata(BaseModel):
    doc_id: str
    filename: str
    page_count: int
    chunk_count: int
    uploaded_at: datetime
    file_size_bytes: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentMetadata]
    total: int


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    page_count: int
    chunk_count: int
    message: str


class DeleteResponse(BaseModel):
    doc_id: str
    message: str


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    role: str                    # "user" | "assistant"
    content: str
    sources: list[SourceChunk] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SourceChunk(BaseModel):
    doc_id: str
    filename: str
    page_number: int | None
    chunk_index: int
    excerpt: str                 # short snippet shown in the UI
    relevance_score: float


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    history: list[dict[str, Any]] = Field(default_factory=list)
    doc_ids: list[str] = Field(
        default_factory=list,
        description="Restrict retrieval to these doc IDs; empty = search all",
    )
    top_k: int = Field(default=5, ge=1, le=20)


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    model_used: str
    tokens_used: int | None = None


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    vector_db_docs: int


# ── Fix forward reference ─────────────────────────────────────────────────────
ChatMessage.model_rebuild()
