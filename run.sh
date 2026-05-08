#!/bin/bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

# ── Prerequisites ──────────────────────────────────────────────
if [ ! -f artifacts/data/raw.csv ]; then
    echo "ERROR: Place your data at artifacts/data/raw.csv first."
    exit 1
fi

if [ -z "${GROQ_API_KEY}" ]; then
    echo "ERROR: GROQ_API_KEY is not set."
    echo "  export GROQ_API_KEY='gsk_...'"
    exit 1
fi

cleanup() {
    echo ""
    echo "Shutting down..."
    kill $LLM_PID $API_PID 2>/dev/null || true
    wait $LLM_PID $API_PID 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM

# ── 1. Start LLM service ──────────────────────────────────────
echo ">>> Starting LLM service..."
export LLM_SERVICE_URL="${LLM_SERVICE_URL:-http://localhost:8001}"
uv run uvicorn llm.app:app --host 0.0.0.0 --port 8001 &
LLM_PID=$!

for i in $(seq 1 30); do
    if curl -sf http://localhost:8001/health > /dev/null 2>&1; then
        echo "LLM service is ready"
        break
    fi
    echo "Waiting for LLM service... attempt $i"
    sleep 1
done

# ── 2. Run training pipeline ───────────────────────────────────
echo ""
echo ">>> Running training pipeline..."
uv run python -m core.train
echo "Training complete."

# ── 3. Start prediction API ────────────────────────────────────
echo ""
echo ">>> Starting prediction API..."
uv run uvicorn api.app:app --host 0.0.0.0 --port 8000 &
API_PID=$!

for i in $(seq 1 30); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "API is ready"
        break
    fi
    echo "Waiting for API... attempt $i"
    sleep 1
done

# ── 4. Start Streamlit UI ──────────────────────────────────────
echo ""
echo ">>> Starting Streamlit UI..."
echo ""
echo "  ┌───────────────────────────────────────────┐"
echo "  │  Open http://localhost:8080 in browser     │"
echo "  │  Ctrl+C to stop everything                 │"
echo "  └───────────────────────────────────────────┘"
echo ""
streamlit run ui/app.py --server.port=${PORT:-8080} --server.address=0.0.0.0
