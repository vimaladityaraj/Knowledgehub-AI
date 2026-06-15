#!/bin/bash
# scripts/docker_entrypoint.sh
# Start FastAPI backend and Streamlit frontend in the same container.
# For production, run each service in its own container (see README).

set -e

echo "Starting KnowledgeHub AI…"

# Start FastAPI in the background
uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!

# Give FastAPI a moment to initialise
sleep 3

# Start Streamlit in the foreground
streamlit run frontend/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false &
STREAMLIT_PID=$!

# Wait for either process to exit
wait -n $FASTAPI_PID $STREAMLIT_PID
