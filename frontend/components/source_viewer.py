"""
frontend/components/source_viewer.py
──────────────────────────────────────
Renders retrieved source chunks as polished citation cards.
"""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_sources(sources: list[dict[str, Any]]) -> None:
    """Display source chunks in an expandable citation section below the answer."""
    if not sources:
        return

    st.markdown(
        """
        <style>
        .source-card {
            border: 1px solid rgba(148, 163, 184, 0.18);
            background: rgba(15, 23, 42, 0.72);
            border-radius: 16px;
            padding: 0.95rem 1rem;
            margin: 0.65rem 0;
        }
        .source-title {
            color: #F8FAFC;
            font-weight: 800;
            font-size: 0.92rem;
            margin-bottom: 0.25rem;
        }
        .source-meta {
            color: #94A3B8;
            font-size: 0.78rem;
            margin-bottom: 0.65rem;
            font-family: 'JetBrains Mono', monospace;
        }
        .source-excerpt {
            color: #CBD5E1;
            line-height: 1.6;
            font-size: 0.9rem;
            border-left: 3px solid rgba(96, 165, 250, 0.75);
            padding-left: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.expander(f"Sources used · {len(sources)} retrieved chunks", expanded=False):
        for i, src in enumerate(sources, start=1):
            page = f"Page {src['page_number']}" if src.get("page_number") else "Page unavailable"
            score = src.get("relevance_score")
            score_text = f" · Relevance {int(score * 100)}%" if isinstance(score, (float, int)) else ""
            filename = src.get("filename", "Unknown file")
            excerpt = src.get("excerpt", "")

            st.markdown(
                f"""
                <div class="source-card">
                    <div class="source-title">Source {i}: {filename}</div>
                    <div class="source-meta">{page}{score_text}</div>
                    <div class="source-excerpt">{excerpt}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
