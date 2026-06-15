"""
frontend/app.py
───────────────
KnowledgeHub AI – polished Streamlit frontend.

Run with:  python -m streamlit run frontend/app.py
"""

from __future__ import annotations

import html

import streamlit as st

from frontend.components.document_panel import render_document_panel
from frontend.components.source_viewer import render_sources
from frontend.utils.api_client import ask_question, health_check

st.set_page_config(
    page_title="KnowledgeHub AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --kh-bg: #0B1020;
        --kh-panel: rgba(15, 23, 42, 0.84);
        --kh-panel-2: rgba(30, 41, 59, 0.72);
        --kh-border: rgba(148, 163, 184, 0.18);
        --kh-text: #E5E7EB;
        --kh-muted: #94A3B8;
        --kh-blue: #60A5FA;
        --kh-purple: #A78BFA;
        --kh-green: #34D399;
        --kh-amber: #FBBF24;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(96, 165, 250, 0.14), transparent 30rem),
            radial-gradient(circle at top right, rgba(167, 139, 250, 0.12), transparent 26rem),
            linear-gradient(180deg, #0B1020 0%, #111827 100%);
        color: var(--kh-text);
    }

    .block-container {
        max-width: 1160px;
        padding-top: 2.1rem;
        padding-bottom: 3rem;
    }

    section[data-testid="stSidebar"] {
        background: rgba(2, 6, 23, 0.86);
        border-right: 1px solid var(--kh-border);
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.6rem;
    }

    .kh-hero {
        position: relative;
        overflow: hidden;
        padding: 1.8rem 1.9rem;
        border: 1px solid var(--kh-border);
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.92), rgba(30, 41, 59, 0.70));
        box-shadow: 0 24px 80px rgba(0, 0, 0, 0.34);
        margin-bottom: 1.35rem;
    }

    .kh-hero::after {
        content: "";
        position: absolute;
        inset: -1px;
        background: radial-gradient(circle at 82% 18%, rgba(96, 165, 250, 0.23), transparent 22rem);
        pointer-events: none;
    }

    .kh-hero-inner {
        position: relative;
        z-index: 1;
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 1.4rem;
    }

    .kh-eyebrow {
        display: inline-flex;
        gap: 0.45rem;
        align-items: center;
        color: #BFDBFE;
        background: rgba(59, 130, 246, 0.12);
        border: 1px solid rgba(96, 165, 250, 0.28);
        padding: 0.32rem 0.7rem;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .kh-hero h1 {
        margin: 0.8rem 0 0.45rem 0;
        font-size: clamp(2.0rem, 4vw, 3.0rem);
        line-height: 1.05;
        font-weight: 800;
        letter-spacing: -0.055em;
        color: #F8FAFC;
    }

    .kh-hero p {
        margin: 0;
        max-width: 760px;
        color: #CBD5E1;
        font-size: 1.02rem;
        line-height: 1.65;
    }

    .kh-status {
        white-space: nowrap;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.55rem 0.85rem;
        border-radius: 999px;
        border: 1px solid var(--kh-border);
        background: rgba(15, 23, 42, 0.78);
        color: var(--kh-muted);
        font-size: 0.86rem;
        font-weight: 700;
    }

    .kh-dot-online { color: var(--kh-green); }
    .kh-dot-offline { color: #FB7185; }

    .kh-stack {
        margin-top: 1.05rem;
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
    }

    .kh-chip {
        border: 1px solid rgba(148, 163, 184, 0.18);
        background: rgba(15, 23, 42, 0.72);
        color: #D1D5DB;
        border-radius: 999px;
        padding: 0.38rem 0.72rem;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .kh-card {
        border: 1px solid var(--kh-border);
        background: rgba(15, 23, 42, 0.72);
        border-radius: 20px;
        padding: 1rem;
        box-shadow: 0 16px 48px rgba(0, 0, 0, 0.18);
    }

    .kh-empty {
        text-align: center;
        padding: 3.4rem 1.4rem;
        border: 1px dashed rgba(148, 163, 184, 0.24);
        border-radius: 24px;
        background: rgba(15, 23, 42, 0.45);
        margin: 1.2rem 0 1.5rem 0;
    }

    .kh-empty-icon {
        width: 72px;
        height: 72px;
        display: inline-grid;
        place-items: center;
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(96, 165, 250, 0.20), rgba(167, 139, 250, 0.20));
        border: 1px solid rgba(148, 163, 184, 0.18);
        font-size: 2rem;
        margin-bottom: 1rem;
    }

    .kh-empty h3 {
        margin: 0 0 0.45rem 0;
        color: #F8FAFC;
        font-size: 1.25rem;
    }

    .kh-empty p {
        margin: 0 auto;
        max-width: 560px;
        color: var(--kh-muted);
        line-height: 1.65;
    }

    .kh-message {
        border-radius: 20px;
        padding: 1rem 1.1rem;
        margin: 0.85rem 0;
        border: 1px solid var(--kh-border);
    }

    .kh-message-user {
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.18), rgba(30, 41, 59, 0.72));
        border-left: 4px solid var(--kh-blue);
    }

    .kh-message-assistant {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12), rgba(30, 41, 59, 0.74));
        border-left: 4px solid var(--kh-green);
    }

    .kh-message-label {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        color: #F8FAFC;
        font-weight: 800;
        font-size: 0.86rem;
        margin-bottom: 0.45rem;
    }

    .kh-message-body {
        color: #D1D5DB;
        line-height: 1.65;
        font-size: 0.96rem;
    }

    .kh-input-wrap {
        margin-top: 1.25rem;
        padding: 1rem;
        border: 1px solid var(--kh-border);
        background: rgba(15, 23, 42, 0.68);
        border-radius: 22px;
    }

    div[data-testid="stTextArea"] textarea {
        min-height: 96px !important;
        border-radius: 16px !important;
        border: 1px solid rgba(148, 163, 184, 0.28) !important;
        background: rgba(2, 6, 23, 0.72) !important;
        color: #F8FAFC !important;
        font-size: 0.98rem !important;
        line-height: 1.55 !important;
    }

    div[data-testid="stTextArea"] textarea:focus {
        border-color: rgba(96, 165, 250, 0.85) !important;
        box-shadow: 0 0 0 4px rgba(96, 165, 250, 0.12) !important;
    }

    .stButton > button {
        border-radius: 14px !important;
        border: 1px solid rgba(148, 163, 184, 0.18) !important;
        font-weight: 800 !important;
        min-height: 2.9rem;
        transition: transform 0.15s ease, border-color 0.15s ease, background 0.15s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        border-color: rgba(96, 165, 250, 0.55) !important;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2563EB, #7C3AED) !important;
        color: #FFFFFF !important;
    }

    div[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2563EB, #4F46E5) !important;
    }

    .stSlider [data-baseweb="slider"] > div > div {
        background-color: #60A5FA !important;
    }

    .stAlert {
        border-radius: 16px !important;
    }

    .kh-small-caption {
        color: var(--kh-muted);
        font-size: 0.82rem;
        line-height: 1.55;
    }

    hr {
        border-color: rgba(148, 163, 184, 0.14) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "top_k" not in st.session_state:
        st.session_state.top_k = 5


_init_state()

selected_doc_ids = render_document_panel()

st.sidebar.markdown("---")
st.sidebar.markdown("### Retrieval settings")
st.session_state.top_k = st.sidebar.slider(
    "Context chunks per answer",
    min_value=1,
    max_value=15,
    value=st.session_state.top_k,
    help="Higher values give the model more context, but may make responses slower.",
)

if st.sidebar.button("Clear chat history", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

health = health_check()
status_label = "Online" if health else "Offline"
status_dot_class = "kh-dot-online" if health else "kh-dot-offline"
status_title = "Backend connected" if health else "Backend unavailable"

st.markdown(
    f"""
    <section class="kh-hero">
        <div class="kh-hero-inner">
            <div>
                <span class="kh-eyebrow">🧠 RAG Assistant</span>
                <h1>KnowledgeHub AI</h1>
                <p>
                    Upload PDFs, retrieve the most relevant passages, and generate grounded answers with source citations.
                    Built as a local-first AI engineering project for document understanding and retrieval-augmented generation.
                </p>
                <div class="kh-stack">
                    <span class="kh-chip">FastAPI backend</span>
                    <span class="kh-chip">Ollama / local LLM</span>
                    <span class="kh-chip">Sentence Transformers</span>
                    <span class="kh-chip">Local vector search</span>
                    <span class="kh-chip">PDF source citations</span>
                </div>
            </div>
            <div class="kh-status" title="{status_title}">
                <span class="{status_dot_class}">●</span> {status_label}
            </div>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

if not st.session_state.messages:
    st.markdown(
        """
        <div class="kh-empty">
            <div class="kh-empty-icon">📚</div>
            <h3>Start by asking a question about your PDFs</h3>
            <p>
                Select one or more indexed documents in the sidebar, then ask a natural-language question.
                KnowledgeHub AI will retrieve the most relevant chunks and cite the sources it used.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    for msg in st.session_state.messages:
        content = html.escape(msg.get("content", "")).replace("\n", "<br>")
        if msg["role"] == "user":
            st.markdown(
                f"""
                <div class="kh-message kh-message-user">
                    <div class="kh-message-label">👤 You</div>
                    <div class="kh-message-body">{content}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="kh-message kh-message-assistant">
                    <div class="kh-message-label">🧠 KnowledgeHub AI</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(msg.get("content", ""))
            if msg.get("sources"):
                render_sources(msg["sources"])

st.markdown('<div class="kh-input-wrap">', unsafe_allow_html=True)
col_input, col_btn = st.columns([0.84, 0.16])

with col_input:
    question = st.text_area(
        "Ask a question",
        placeholder="Example: What is the main contribution of this document?",
        height=100,
        label_visibility="collapsed",
        key="question_input",
    )

with col_btn:
    send = st.button("Send →", type="primary", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

if send and question.strip():
    if not health:
        st.error("Cannot reach the FastAPI backend. Start it with: python -m uvicorn backend.main:app --reload --port 8000")
    elif not selected_doc_ids:
        st.warning("Select at least one indexed document in the sidebar before asking a question.")
    else:
        st.session_state.messages.append({"role": "user", "content": question.strip()})
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ]

        with st.spinner("Retrieving sources and generating an answer..."):
            try:
                response = ask_question(
                    question=question.strip(),
                    history=history,
                    doc_ids=selected_doc_ids,
                    top_k=st.session_state.top_k,
                )
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response.get("sources", []),
                    }
                )
                if response.get("tokens_used"):
                    st.toast(
                        f"Model: {response.get('model_used', 'unknown')} · {response['tokens_used']} tokens",
                        icon="📊",
                    )
            except Exception as exc:
                st.session_state.messages.pop()
                st.error(f"Answer generation failed: {exc}")

        st.rerun()
