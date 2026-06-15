"""
backend/utils/chunker.py
────────────────────────
Splits extracted document text into overlapping chunks suitable for embedding.

Pure-Python implementation — zero external dependencies.

Algorithm (recursive separator splitting):
  1. Try to split on paragraph breaks, then newlines, then sentences,
     then words, falling back to raw characters if needed.
  2. Greedily merge small pieces into chunks of at most `chunk_size` chars.
  3. Carry forward `chunk_overlap` chars from the previous chunk so context
     is not lost at boundaries.
  4. Tiny tail chunks (< 100 chars) are merged into the preceding chunk.
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from backend.utils.pdf_extractor import ExtractedDocument

# Separators tried in order — broadest to narrowest
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class TextChunk:
    chunk_index: int          # global index within the document
    text: str
    page_number: int | None   # source page (best-effort)
    doc_id: str
    filename: str


# ── Core splitting logic ───────────────────────────────────────────────────────

def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Recursively split *text* into chunks of at most *chunk_size* characters
    with *chunk_overlap* characters of context carried between chunks.
    """
    # Step 1 – find the first separator that actually divides the text
    pieces: list[str] = [text]
    for sep in _SEPARATORS:
        if sep and sep in text:
            pieces = text.split(sep)
            break
        if not sep:
            # Last resort: split into individual characters
            pieces = list(text)

    # Step 2 – greedily merge pieces into chunks
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for piece in pieces:
        piece_len = len(piece)

        if current_len + piece_len > chunk_size and current:
            # Flush current chunk
            chunk_text = " ".join(current).strip()
            if chunk_text:
                chunks.append(chunk_text)

            # Seed next chunk with overlap from the tail of current
            overlap_chars = 0
            overlap_pieces: list[str] = []
            for p in reversed(current):
                if overlap_chars + len(p) <= chunk_overlap:
                    overlap_pieces.insert(0, p)
                    overlap_chars += len(p)
                else:
                    break
            current = overlap_pieces
            current_len = overlap_chars

        current.append(piece)
        current_len += piece_len

    # Flush final chunk
    if current:
        tail = " ".join(current).strip()
        if tail:
            chunks.append(tail)

    return chunks


# ── Public API ─────────────────────────────────────────────────────────────────

def chunk_document(
    doc: ExtractedDocument,
    doc_id: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[TextChunk]:
    """
    Chunk an ExtractedDocument into overlapping TextChunks.

    Strategy:
      - Process each page independently to preserve page-number metadata.
      - Merge tiny tail chunks (< 100 chars) into the previous chunk to avoid
        near-empty embeddings.
    """
    chunks: list[TextChunk] = []
    global_idx = 0

    for page in doc.pages:
        if not page.text.strip():
            continue

        raw_chunks = _split_text(page.text, chunk_size, chunk_overlap)

        for raw in raw_chunks:
            cleaned = raw.strip()
            if not cleaned:
                continue

            # Absorb tiny tail fragments into the previous chunk
            if len(cleaned) < 100 and chunks:
                chunks[-1].text += " " + cleaned
                continue

            chunks.append(
                TextChunk(
                    chunk_index=global_idx,
                    text=cleaned,
                    page_number=page.page_number,
                    doc_id=doc_id,
                    filename=doc.filename,
                )
            )
            global_idx += 1

    logger.info(
        f"Chunked '{doc.filename}' → {len(chunks)} chunks "
        f"(size={chunk_size}, overlap={chunk_overlap})"
    )
    return chunks
