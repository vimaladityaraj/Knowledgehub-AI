"""
backend/api/documents.py
────────────────────────
FastAPI router for document upload, listing, and deletion.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from loguru import logger

from backend.core.config import get_settings
from backend.db.document_store import get_document_store
from backend.db.vector_store import get_vector_store
from backend.models.schemas import (
    DeleteResponse,
    DocumentListResponse,
    DocumentMetadata,
    UploadResponse,
)
from backend.utils.chunker import chunk_document
from backend.utils.pdf_extractor import extract_text_from_bytes

router = APIRouter(prefix="/documents", tags=["Documents"])

_MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """
    Upload a PDF document.

    1. Validate the file.
    2. Extract text page-by-page.
    3. Chunk the text.
    4. Embed chunks and upsert into ChromaDB.
    5. Persist metadata to the document store.
    """
    cfg = get_settings()

    # ── Validate ──────────────────────────────────────────────────────────────
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are accepted.",
        )

    raw_bytes = await file.read()
    if len(raw_bytes) > _MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size of {_MAX_FILE_SIZE // (1024*1024)} MB.",
        )
    if len(raw_bytes) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file.")

    doc_id = str(uuid.uuid4())
    filename = file.filename

    try:
        # ── Extract text ──────────────────────────────────────────────────────
        extracted = extract_text_from_bytes(raw_bytes, filename)
        if extracted.page_count == 0 or extracted.total_chars < 10:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Could not extract readable text from the PDF.",
            )

        # ── Save file to disk ─────────────────────────────────────────────────
        save_path: Path = cfg.upload_path / f"{doc_id}_{filename}"
        save_path.write_bytes(raw_bytes)

        # ── Chunk ─────────────────────────────────────────────────────────────
        chunks = chunk_document(
            doc=extracted,
            doc_id=doc_id,
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
        )

        # ── Embed & upsert ────────────────────────────────────────────────────
        get_vector_store().upsert_chunks(chunks)

        # ── Persist metadata ──────────────────────────────────────────────────
        get_document_store().add(
            doc_id=doc_id,
            filename=filename,
            page_count=extracted.page_count,
            chunk_count=len(chunks),
            file_size_bytes=len(raw_bytes),
        )

        logger.info(f"Upload complete: '{filename}' → doc_id={doc_id}")
        return UploadResponse(
            doc_id=doc_id,
            filename=filename,
            page_count=extracted.page_count,
            chunk_count=len(chunks),
            message="Document processed and indexed successfully.",
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"Upload failed for '{filename}': {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed: {str(exc)}",
        )


@router.get("/", response_model=DocumentListResponse)
async def list_documents() -> DocumentListResponse:
    """Return all indexed documents with their metadata."""
    docs_raw = get_document_store().list_all()
    docs = [
        DocumentMetadata(
            doc_id=d["doc_id"],
            filename=d["filename"],
            page_count=d["page_count"],
            chunk_count=d["chunk_count"],
            uploaded_at=datetime.fromisoformat(d["uploaded_at"]),
            file_size_bytes=d["file_size_bytes"],
        )
        for d in docs_raw
    ]
    return DocumentListResponse(documents=docs, total=len(docs))


@router.delete("/{doc_id}", response_model=DeleteResponse)
async def delete_document(doc_id: str) -> DeleteResponse:
    """Remove a document and all its chunks from the system."""
    store = get_document_store()
    if not store.exists(doc_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found.",
        )

    # Delete from vector DB
    deleted_chunks = get_vector_store().delete_document(doc_id)

    # Delete saved PDF file
    cfg = get_settings()
    for pdf_file in cfg.upload_path.glob(f"{doc_id}_*"):
        pdf_file.unlink(missing_ok=True)

    # Delete metadata
    store.delete(doc_id)

    logger.info(f"Deleted document {doc_id} ({deleted_chunks} chunks removed)")
    return DeleteResponse(doc_id=doc_id, message="Document deleted successfully.")
