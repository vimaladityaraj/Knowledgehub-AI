"""
backend/db/document_store.py
─────────────────────────────
Lightweight JSON-backed store for document metadata (filename, page count, etc.).
This complements the ChromaDB vector store which only holds chunk data.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

from loguru import logger

from backend.core.config import get_settings

_STORE_FILE = "doc_metadata.json"


class DocumentStore:
    """Thread-safe JSON file store for document metadata."""

    def __init__(self) -> None:
        cfg = get_settings()
        self._path = cfg.upload_path / _STORE_FILE
        self._lock = Lock()
        self._data: dict[str, dict[str, Any]] = self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> dict[str, dict[str, Any]]:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning(f"Could not read doc store: {exc}")
        return {}

    def _save(self) -> None:
        self._path.write_text(
            json.dumps(self._data, indent=2, default=str),
            encoding="utf-8",
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def add(
        self,
        doc_id: str,
        filename: str,
        page_count: int,
        chunk_count: int,
        file_size_bytes: int,
    ) -> None:
        with self._lock:
            self._data[doc_id] = {
                "doc_id": doc_id,
                "filename": filename,
                "page_count": page_count,
                "chunk_count": chunk_count,
                "file_size_bytes": file_size_bytes,
                "uploaded_at": datetime.utcnow().isoformat(),
            }
            self._save()
        logger.info(f"DocumentStore: added '{filename}' ({doc_id})")

    def get(self, doc_id: str) -> dict[str, Any] | None:
        return self._data.get(doc_id)

    def delete(self, doc_id: str) -> bool:
        with self._lock:
            if doc_id in self._data:
                del self._data[doc_id]
                self._save()
                return True
        return False

    def list_all(self) -> list[dict[str, Any]]:
        return list(self._data.values())

    def exists(self, doc_id: str) -> bool:
        return doc_id in self._data


from functools import lru_cache


@lru_cache(maxsize=1)
def get_document_store() -> DocumentStore:
    return DocumentStore()
