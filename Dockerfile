# ══════════════════════════════════════════════════════════════════════════════
# KnowledgeHub AI – Multi-stage Dockerfile
# ══════════════════════════════════════════════════════════════════════════════

# ── Base ──────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

# System dependencies for PyMuPDF and other native libs
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libgl1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Dependencies ──────────────────────────────────────────────────────────────
FROM base AS deps

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model so it's baked into the image
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# ── App ───────────────────────────────────────────────────────────────────────
FROM deps AS app

COPY . .

# Ensure data directories exist
RUN mkdir -p data/uploads data/chroma

# Copy environment template (actual secrets must be provided at runtime)
RUN cp .env.example .env

EXPOSE 8000 8501

# ── Entrypoint ────────────────────────────────────────────────────────────────
# Use a simple shell script to start both services.
# For production, prefer separate containers via docker-compose.
COPY scripts/docker_entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
