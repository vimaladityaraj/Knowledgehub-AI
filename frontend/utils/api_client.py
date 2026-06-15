"""
frontend/utils/api_client.py
─────────────────────────────
Thin HTTP client wrapping the FastAPI backend.
All network calls go through here so the Streamlit pages stay clean.
"""

from __future__ import annotations

import os
from typing import Any, BinaryIO

import requests
from loguru import logger

_BASE_URL = os.getenv("STREAMLIT_API_BASE_URL", "http://localhost:8000")
_TIMEOUT = 120  # seconds – generous for large PDFs / slow LLM responses


def _url(path: str) -> str:
    return f"{_BASE_URL.rstrip('/')}{path}"


# ── Documents ──────────────────────────────────────────────────────────────────

def upload_pdf(file_obj: BinaryIO, filename: str) -> dict[str, Any]:
    """Upload a PDF to the backend and return the response JSON."""
    resp = requests.post(
        _url("/documents/upload"),
        files={"file": (filename, file_obj, "application/pdf")},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def list_documents() -> list[dict[str, Any]]:
    resp = requests.get(_url("/documents/"), timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json().get("documents", [])


def delete_document(doc_id: str) -> dict[str, Any]:
    resp = requests.delete(_url(f"/documents/{doc_id}"), timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


# ── Chat ───────────────────────────────────────────────────────────────────────

def ask_question(
    question: str,
    history: list[dict[str, Any]],
    doc_ids: list[str] | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    payload = {
        "question": question,
        "history": history,
        "doc_ids": doc_ids or [],
        "top_k": top_k,
    }
    resp = requests.post(_url("/chat/"), json=payload, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


# ── Health ─────────────────────────────────────────────────────────────────────

def health_check() -> dict[str, Any] | None:
    try:
        resp = requests.get(_url("/health/"), timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning(f"Health check failed: {exc}")
        return None
