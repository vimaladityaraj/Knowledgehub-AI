"""
backend/core/llm_client.py
──────────────────────────
Unified LLM client supporting three providers:
  • anthropic  – Anthropic Claude via the official SDK
  • openai     – OpenAI GPT via the official SDK
  • ollama     – any locally-running Ollama model via its REST API
                 (no extra SDK needed – uses the stdlib `urllib` / requests)

Swap providers by setting LLM_PROVIDER in your .env file.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from functools import lru_cache
from typing import Any

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.core.config import get_settings

# ── System prompt ─────────────────────────────────────────────────────────────
_RAG_SYSTEM_PROMPT = """\
You are KnowledgeHub AI, a precise and helpful research assistant.
Answer the user's question ONLY using the provided context excerpts.
Rules:
  1. Be concise but thorough.
  2. Cite sources by referencing [Source N] inline where appropriate.
  3. If the context does not contain enough information, say so clearly.
  4. Never fabricate facts or cite sources that are not provided.
  5. Use markdown formatting (bold, lists, headers) to improve readability.
"""

# Models known to use Ollama's /api/chat endpoint correctly.
# All current Ollama models support the chat endpoint, but we keep this
# list for documentation / validation purposes.
_OLLAMA_CHAT_MODELS = {
    "qwen3:8b", "qwen3:14b", "qwen3:32b",
    "llama3", "llama3:8b", "llama3:70b",
    "llama3.1", "llama3.1:8b", "llama3.1:70b",
    "llama3.2", "llama3.2:3b",
    "mistral", "mistral:7b",
    "phi3", "phi3:mini", "phi3:medium",
    "gemma2", "gemma2:2b", "gemma2:9b", "gemma2:27b",
    "deepseek-r1", "deepseek-r1:7b",
    "codellama", "codellama:7b",
}


class LLMClient:
    """
    Provider-agnostic LLM wrapper.

    Usage:
        client = LLMClient()
        answer, tokens = client.answer(question, context_chunks, history)
    """

    def __init__(self) -> None:
        cfg = get_settings()
        self._provider = cfg.llm_provider.lower()
        self._model    = cfg.active_model   # resolved by config property
        self._cfg      = cfg

        if self._provider not in ("anthropic", "openai", "ollama"):
            raise ValueError(
                f"Unknown LLM_PROVIDER={self._provider!r}. "
                "Choose 'anthropic', 'openai', or 'ollama'."
            )

        logger.info(
            f"LLMClient ready — provider={self._provider}, model={self._model}"
            + (f", base_url={cfg.ollama_base_url}" if self._provider == "ollama" else "")
        )

    # ── Public API ────────────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def answer(
        self,
        question: str,
        context_chunks: list[dict[str, Any]],
        history: list[dict[str, Any]] | None = None,
    ) -> tuple[str, int | None]:
        """
        Generate a RAG answer grounded in *context_chunks*.

        Args:
            question:       The user's natural-language question.
            context_chunks: Retrieved chunks from the vector store.
            history:        Previous conversation turns (role/content dicts).

        Returns:
            (answer_text, tokens_used)  — tokens_used is None for Ollama
            because the /api/chat endpoint does not always expose token counts
            in a consistent way across models.
        """
        context_str  = self._build_context(context_chunks)
        user_message = (
            f"Context excerpts:\n{context_str}\n\n"
            f"Question: {question}"
        )

        dispatch = {
            "anthropic": self._call_anthropic,
            "openai":    self._call_openai,
            "ollama":    self._call_ollama,
        }
        return dispatch[self._provider](user_message, history or [])

    # ── Anthropic ─────────────────────────────────────────────────────────────

    def _call_anthropic(
        self,
        user_message: str,
        history: list[dict[str, Any]],
    ) -> tuple[str, int]:
        import anthropic

        client   = anthropic.Anthropic(api_key=self._cfg.anthropic_api_key)
        messages = self._build_messages(user_message, history, provider="anthropic")

        response = client.messages.create(
            model=self._model,
            max_tokens=1500,
            system=_RAG_SYSTEM_PROMPT,
            messages=messages,
        )
        text   = response.content[0].text
        tokens = response.usage.input_tokens + response.usage.output_tokens
        return text, tokens

    # ── OpenAI ────────────────────────────────────────────────────────────────

    def _call_openai(
        self,
        user_message: str,
        history: list[dict[str, Any]],
    ) -> tuple[str, int | None]:
        from openai import OpenAI

        client   = OpenAI(api_key=self._cfg.openai_api_key)
        messages = [{"role": "system", "content": _RAG_SYSTEM_PROMPT}]
        messages += self._build_messages(user_message, history, provider="openai")

        response = client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=1500,
            temperature=0.2,
        )
        text   = response.choices[0].message.content or ""
        tokens = response.usage.total_tokens if response.usage else None
        return text, tokens

    # ── Ollama ────────────────────────────────────────────────────────────────

    def _call_ollama(
        self,
        user_message: str,
        history: list[dict[str, Any]],
    ) -> tuple[str, None]:
        """
        Call the Ollama /api/chat endpoint.

        Ollama's chat API mirrors the OpenAI chat format:
          POST /api/chat
          { "model": "...", "messages": [...], "stream": false }

        No extra SDK is required — we use Python's built-in urllib so the
        project stays pip-installable on Windows without any native build step.
        """
        base_url = self._cfg.ollama_base_url.rstrip("/")
        url      = f"{base_url}/api/chat"

        # Prepend system turn
        messages: list[dict[str, str]] = [
            {"role": "system", "content": _RAG_SYSTEM_PROMPT}
        ]
        messages += self._build_messages(user_message, history, provider="ollama")

        payload = json.dumps({
            "model":    self._model,
            "messages": messages,
            "stream":   False,          # get the full response in one shot
            "options": {
                "temperature": 0.2,
                "num_predict": 1500,    # max tokens to generate
            },
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"Could not reach Ollama at {base_url}. "
                "Make sure Ollama is running (`ollama serve`) and "
                f"OLLAMA_BASE_URL is correct. Original error: {exc}"
            ) from exc

        # Ollama non-streaming response shape:
        # { "message": { "role": "assistant", "content": "..." }, ... }
        try:
            text = body["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"Unexpected Ollama response shape: {body}"
            ) from exc

        # Report eval token count if available (Ollama ≥ 0.1.38)
        tokens_generated = body.get("eval_count")
        if tokens_generated:
            logger.debug(f"Ollama eval_count={tokens_generated}")

        return text, None   # token count not standardised across models

    # ── Shared helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _build_context(chunks: list[dict[str, Any]]) -> str:
        """Format retrieved chunks into a numbered context block."""
        parts: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            page = f", page {chunk['page_number']}" if chunk.get("page_number") else ""
            parts.append(f"[Source {i}] {chunk['filename']}{page}\n{chunk['text']}")
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _build_messages(
        user_message: str,
        history: list[dict[str, Any]],
        provider: str,
    ) -> list[dict[str, Any]]:
        """
        Convert the flat history list + new user message into the
        role/content format expected by all three providers.

        Keeps the last 6 turns to avoid blowing the context window.
        """
        messages: list[dict[str, Any]] = []
        for turn in history[-6:]:
            role    = turn.get("role", "user")
            content = turn.get("content", "")
            # Anthropic only accepts "user" and "assistant" roles
            if provider == "anthropic" and role not in ("user", "assistant"):
                continue
            messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_message})
        return messages


@lru_cache(maxsize=1)
def get_llm_client() -> LLMClient:
    """Return a process-level singleton LLMClient."""
    return LLMClient()
