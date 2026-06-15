"""
backend/db/vector_store.py
──────────────────────────
Pure-NumPy vector store — no C++ build tools required, works on Windows.

Storage layout  (all under `data/vectors/`):
  vectors.npy      float32 matrix, shape (N, D)
  metadata.json    list of chunk records (doc_id, filename, page, chunk_index, text)

The index is loaded entirely into memory on startup and flushed to disk after
every mutating operation.  For typical RAG workloads (tens of thousands of
chunks) this is fast enough; NumPy cosine similarity over 50 k × 384-dim
vectors completes in < 50 ms on a modern laptop.

Public interface is identical to the old ChromaDB wrapper so no other file
needs to change.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from backend.core.config import get_settings
from backend.utils.chunker import TextChunk

# ── Disk filenames ─────────────────────────────────────────────────────────────
_VECTORS_FILE  = "vectors.npy"
_METADATA_FILE = "metadata.json"


class VectorStore:
    """
    In-memory NumPy vector store with JSON-backed persistence.

    Thread-safety: a single RLock guards both the in-memory arrays and all
    disk writes so concurrent FastAPI requests cannot corrupt state.
    """

    def __init__(self) -> None:
        cfg = get_settings()
        self._store_dir: Path = cfg.vector_path   # data/vectors/
        self._lock = Lock()
        self._embedder = SentenceTransformer(cfg.embedding_model)

        # In-memory state
        self._vectors: np.ndarray  # shape (N, D), float32
        self._meta: list[dict[str, Any]]  # parallel list of chunk records

        self._load()
        logger.info(
            f"VectorStore ready – {len(self._meta)} chunks in memory "
            f"(store: {self._store_dir})"
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load vectors + metadata from disk, or initialise empty store."""
        vec_path  = self._store_dir / _VECTORS_FILE
        meta_path = self._store_dir / _METADATA_FILE

        if vec_path.exists() and meta_path.exists():
            try:
                self._vectors = np.load(str(vec_path))
                self._meta    = json.loads(meta_path.read_text(encoding="utf-8"))
                logger.debug(f"Loaded {len(self._meta)} chunks from disk.")
                return
            except Exception as exc:
                logger.warning(f"Could not load vector store from disk: {exc}. Starting fresh.")

        # Empty store
        self._vectors = np.empty((0, 0), dtype=np.float32)
        self._meta    = []

    def _save(self) -> None:
        """Flush in-memory state to disk. Must be called inside self._lock."""
        vec_path  = self._store_dir / _VECTORS_FILE
        meta_path = self._store_dir / _METADATA_FILE
        np.save(str(vec_path), self._vectors)
        meta_path.write_text(
            json.dumps(self._meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ── Embedding helper ──────────────────────────────────────────────────────

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalised float32 embeddings, shape (len(texts), D)."""
        vecs = self._embedder.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,   # unit vectors → dot == cosine similarity
        ).astype(np.float32)
        return vecs

    # ── Public API ────────────────────────────────────────────────────────────

    def upsert_chunks(self, chunks: list[TextChunk]) -> None:
        """Embed *chunks* and add them to the store, replacing any existing
        chunk with the same (doc_id, chunk_index) key."""
        if not chunks:
            return

        new_vecs = self._embed([c.text for c in chunks])
        new_meta = [
            {
                "doc_id":      c.doc_id,
                "filename":    c.filename,
                "page_number": c.page_number,  # None is JSON-serialisable
                "chunk_index": c.chunk_index,
                "text":        c.text,
            }
            for c in chunks
        ]

        with self._lock:
            # Build a lookup of existing positions keyed by (doc_id, chunk_index)
            existing_keys: dict[tuple[str, int], int] = {
                (m["doc_id"], m["chunk_index"]): i
                for i, m in enumerate(self._meta)
            }

            add_vecs:  list[np.ndarray]       = []
            add_meta:  list[dict[str, Any]]   = []

            for vec, meta in zip(new_vecs, new_meta):
                key = (meta["doc_id"], meta["chunk_index"])
                if key in existing_keys:
                    # Update in-place
                    idx = existing_keys[key]
                    self._vectors[idx] = vec
                    self._meta[idx]    = meta
                else:
                    add_vecs.append(vec)
                    add_meta.append(meta)

            if add_vecs:
                new_block = np.stack(add_vecs, axis=0)
                if self._vectors.size == 0:
                    self._vectors = new_block
                else:
                    self._vectors = np.concatenate([self._vectors, new_block], axis=0)
                self._meta.extend(add_meta)

            self._save()

        logger.info(f"Upserted {len(chunks)} chunks for doc_id='{chunks[0].doc_id}'")

    def query(
        self,
        question: str,
        top_k: int = 5,
        doc_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return the *top_k* most relevant chunks for *question*.

        Args:
            question:  Natural-language query string.
            top_k:     Maximum number of results.
            doc_ids:   If provided, restrict search to these document IDs.

        Returns:
            List of dicts with keys: doc_id, filename, page_number,
            chunk_index, text, score.  Sorted by score descending.
        """
        with self._lock:
            if len(self._meta) == 0:
                return []

            # Optionally filter to a subset of documents
            if doc_ids:
                doc_id_set = set(doc_ids)
                indices = [i for i, m in enumerate(self._meta) if m["doc_id"] in doc_id_set]
            else:
                indices = list(range(len(self._meta)))

            if not indices:
                return []

            subset_vecs = self._vectors[indices]   # (M, D)
            subset_meta = [self._meta[i] for i in indices]

        # Embed query (outside lock – encoding can be slow)
        q_vec = self._embed([question])[0]  # (D,)

        # Cosine similarity = dot product of unit vectors
        scores: np.ndarray = subset_vecs @ q_vec  # (M,)

        k = min(top_k, len(scores))
        # argpartition is O(M) vs O(M log M) for argsort – fast for large M
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results: list[dict[str, Any]] = []
        for idx in top_indices:
            meta  = subset_meta[idx]
            score = float(scores[idx])
            results.append(
                {
                    "doc_id":      meta["doc_id"],
                    "filename":    meta["filename"],
                    "page_number": meta["page_number"],
                    "chunk_index": meta["chunk_index"],
                    "text":        meta["text"],
                    "score":       round(score, 4),
                }
            )
        return results

    def delete_document(self, doc_id: str) -> int:
        """Remove every chunk belonging to *doc_id*. Returns count deleted."""
        with self._lock:
            keep = [i for i, m in enumerate(self._meta) if m["doc_id"] != doc_id]
            deleted = len(self._meta) - len(keep)

            if deleted:
                self._vectors = self._vectors[keep] if keep else np.empty((0, 0), dtype=np.float32)
                self._meta    = [self._meta[i] for i in keep]
                self._save()

        logger.info(f"Deleted {deleted} chunks for doc_id='{doc_id}'")
        return deleted

    def list_documents(self) -> list[dict[str, Any]]:
        """Deduplicated list of {doc_id, filename, chunk_count} records."""
        with self._lock:
            seen: dict[str, dict[str, Any]] = {}
            for m in self._meta:
                doc_id = m["doc_id"]
                if doc_id not in seen:
                    seen[doc_id] = {
                        "doc_id":      doc_id,
                        "filename":    m["filename"],
                        "chunk_count": 0,
                    }
                seen[doc_id]["chunk_count"] += 1
        return list(seen.values())

    @property
    def total_chunks(self) -> int:
        with self._lock:
            return len(self._meta)


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    """Return a process-level singleton VectorStore."""
    return VectorStore()
