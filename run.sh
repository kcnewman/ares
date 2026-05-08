#!/bin/bash
set -e

if [ ! -f data/raw.csv ]; then
    echo "Error: data/raw.csv not found"
    exit 1
fi

if [ ! -f models/model.joblib ]; then
    echo "No model found. Run 'uv run python -m lib.train' first."
    exit 1
fi

echo "Starting API on :8000..."
uvicorn api.server:app --host 0.0.0.0 --port 8000 &
API_PID=$!

sleep 2

echo "Starting UI on :8080..."
streamlit run ui/app.py --server.port=8080 --server.address=0.0.0.0

trap "kill $API_PID 2>/dev/null" EXIT
