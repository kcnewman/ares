#!/bin/bash
set -e

echo "Starting API server on port 8000..."
uvicorn api.server:app --host 0.0.0.0 --port 8000 2>&1 &

echo "Starting Streamlit UI on port ${PORT:-8080}..."
exec streamlit run ui/app.py --server.port=${PORT:-8080} --server.address=0.0.0.0
