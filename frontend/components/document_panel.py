"""
frontend/components/document_panel.py
──────────────────────────────────────
Polished Streamlit sidebar component for uploading and managing documents.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from frontend.utils.api_client import delete_document, list_documents, upload_pdf


def _short_name(filename: str, max_len: int = 34) -> str:
    if len(filename) <= max_len:
        return filename
    stem, dot, suffix = filename.rpartition(".")
    shortened = stem[: max_len - len(suffix) - 5].rstrip()
    return f"{shortened}…{dot}{suffix}" if dot else f"{filename[:max_len-1]}…"


def render_document_panel() -> list[str]:
    """
    Render the document management panel in the sidebar.

    Returns a list of doc_ids that are currently selected for retrieval.
    """
    st.sidebar.markdown(
        """
        <div style="padding:0.2rem 0 1rem 0;">
            <div style="font-size:1.45rem;font-weight:800;color:#F8FAFC;letter-spacing:-0.04em;">KnowledgeHub AI</div>
            <div style="color:#94A3B8;font-size:0.86rem;line-height:1.5;margin-top:0.25rem;">
                Local document search with grounded answers.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("### Upload document")
    uploaded_file = st.sidebar.file_uploader(
        "PDF upload",
        type=["pdf"],
        help="Upload a PDF and index it for retrieval.",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        st.sidebar.caption(f"Selected: `{_short_name(uploaded_file.name, 42)}`")
        if st.sidebar.button("Index document", use_container_width=True, type="primary"):
            with st.spinner(f"Indexing {uploaded_file.name}..."):
                try:
                    result = upload_pdf(uploaded_file, uploaded_file.name)
                    st.sidebar.success(
                        f"Indexed {result['filename']} · {result['page_count']} pages · {result['chunk_count']} chunks"
                    )
                    st.rerun()
                except Exception as exc:
                    st.sidebar.error(f"Upload failed: {exc}")
    else:
        st.sidebar.info("Upload a PDF to add it to your searchable knowledge base.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Indexed documents")

    try:
        docs: list[dict[str, Any]] = list_documents()
    except Exception:
        st.sidebar.error("Cannot reach the backend API.")
        return []

    if not docs:
        st.sidebar.markdown(
            """
            <div style="border:1px dashed rgba(148,163,184,0.25);border-radius:16px;padding:1rem;color:#94A3B8;font-size:0.9rem;line-height:1.55;">
                No documents indexed yet. Once you upload a PDF, it will appear here.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return []

    selected_ids: list[str] = []

    for doc in docs:
        filename = _short_name(doc["filename"], 30)
        with st.sidebar.container(border=True):
            checked = st.checkbox(
                filename,
                value=True,
                key=f"doc_select_{doc['doc_id']}",
                help=doc["filename"],
            )
            st.caption(f"{doc['page_count']} pages · {doc['chunk_count']} chunks")

            col_a, col_b = st.columns([0.62, 0.38])
            with col_a:
                if checked:
                    selected_ids.append(doc["doc_id"])
                    st.markdown("<span style='color:#34D399;font-size:0.82rem;'>● Included</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span style='color:#94A3B8;font-size:0.82rem;'>○ Not selected</span>", unsafe_allow_html=True)
            with col_b:
                if st.button("Delete", key=f"del_{doc['doc_id']}", help="Delete document", use_container_width=True):
                    try:
                        delete_document(doc["doc_id"])
                        st.toast(f"Deleted {doc['filename']}", icon="🗑️")
                        st.rerun()
                    except Exception as exc:
                        st.sidebar.error(f"Delete failed: {exc}")

    st.sidebar.markdown("---")
    if selected_ids:
        st.sidebar.success(f"Searching {len(selected_ids)} selected document(s).")
    else:
        st.sidebar.warning("Select at least one document to search.")

    return selected_ids
