"""
backend/api/health.py
─────────────────────
Simple liveness / readiness endpoint.
"""

from fastapi import APIRouter

from backend.db.vector_store import get_vector_store
from backend.models.schemas import HealthResponse

router = APIRouter(prefix="/health", tags=["Health"])

_VERSION = "1.0.0"


@router.get("/", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return service health and basic stats."""
    try:
        chunk_count = get_vector_store().total_chunks
    except Exception:
        chunk_count = -1

    return HealthResponse(
        status="ok",
        version=_VERSION,
        vector_db_docs=chunk_count,
    )
