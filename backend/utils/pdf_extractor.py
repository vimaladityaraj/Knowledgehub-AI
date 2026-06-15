"""
backend/utils/pdf_extractor.py
───────────────────────────────
Extracts raw text (with page numbers) from uploaded PDF files.
Uses PyMuPDF (fitz) as the primary engine and pdfplumber as fallback.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger


@dataclass
class PageText:
    page_number: int          # 1-based
    text: str
    char_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.char_count = len(self.text)


@dataclass
class ExtractedDocument:
    filename: str
    pages: list[PageText]

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip())

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def total_chars(self) -> int:
        return sum(p.char_count for p in self.pages)


def extract_text_from_bytes(file_bytes: bytes, filename: str) -> ExtractedDocument:
    """
    Extract per-page text from a PDF supplied as raw bytes.
    Tries PyMuPDF first; falls back to pdfplumber if extraction is poor.
    """
    pages = _try_pymupdf(file_bytes)

    # If PyMuPDF yielded very little text, retry with pdfplumber
    total_chars = sum(len(p.text) for p in pages)
    if total_chars < 100 and len(pages) > 0:
        logger.warning(
            f"{filename}: PyMuPDF returned sparse text ({total_chars} chars), "
            "falling back to pdfplumber."
        )
        pages = _try_pdfplumber(file_bytes)

    doc = ExtractedDocument(filename=filename, pages=pages)
    logger.info(
        f"Extracted {doc.page_count} pages / {doc.total_chars} chars from '{filename}'"
    )
    return doc


def extract_text_from_path(pdf_path: Path) -> ExtractedDocument:
    """Convenience wrapper that reads the file from disk first."""
    data = pdf_path.read_bytes()
    return extract_text_from_bytes(data, pdf_path.name)


# ── Private helpers ───────────────────────────────────────────────────────────

def _try_pymupdf(file_bytes: bytes) -> list[PageText]:
    """Use PyMuPDF (fitz) to extract text page by page."""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages: list[PageText] = []
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text")  # plain text, preserves layout
            pages.append(PageText(page_number=i, text=text.strip()))
        doc.close()
        return pages
    except Exception as exc:
        logger.error(f"PyMuPDF extraction failed: {exc}")
        return []


def _try_pdfplumber(file_bytes: bytes) -> list[PageText]:
    """Use pdfplumber to extract text page by page."""
    try:
        import pdfplumber

        pages: list[PageText] = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                pages.append(PageText(page_number=i, text=text.strip()))
        return pages
    except Exception as exc:
        logger.error(f"pdfplumber extraction failed: {exc}")
        return []
