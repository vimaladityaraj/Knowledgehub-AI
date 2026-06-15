"""
backend/core/config.py
──────────────────────
Centralised configuration loaded from environment variables / .env file.
Supported LLM providers: "anthropic" | "openai" | "ollama"
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All tuneable knobs for the application."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM provider ─────────────────────────────────────────────────────────
    # Accepted values: "anthropic" | "openai" | "ollama"
    llm_provider: str = "anthropic"

    # ── Anthropic ─────────────────────────────────────────────────────────────
    anthropic_api_key: str = ""
    # Default model when provider=anthropic
    llm_model: str = "claude-3-5-sonnet-20241022"

    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str = ""
    # Override llm_model in .env, e.g. LLM_MODEL=gpt-4o

    # ── Ollama ────────────────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"
    # ollama_model can also be: llama3, mistral, phi3, gemma2, etc.
    # When provider=ollama, ollama_model takes precedence over llm_model.

    # ── Embeddings ───────────────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"

    # ── Chunking ─────────────────────────────────────────────────────────────
    chunk_size: int = 800
    chunk_overlap: int = 150

    # ── Retrieval ────────────────────────────────────────────────────────────
    top_k_results: int = 5

    # ── Paths ────────────────────────────────────────────────────────────────
    upload_dir: str = "data/uploads"
    vector_dir: str = "data/vectors"

    # ── Server ───────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ── Frontend ─────────────────────────────────────────────────────────────
    streamlit_api_base_url: str = "http://localhost:8000"

    # ── Derived helpers ───────────────────────────────────────────────────────

    @property
    def active_model(self) -> str:
        """Return the model name that will actually be used at runtime."""
        if self.llm_provider.lower() == "ollama":
            return self.ollama_model
        return self.llm_model

    @property
    def upload_path(self) -> Path:
        p = Path(self.upload_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def vector_path(self) -> Path:
        p = Path(self.vector_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings singleton."""
    return Settings()
