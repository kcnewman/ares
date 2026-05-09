#!/bin/bash
set -e

echo "=== Diagnostics ==="
echo "Python: $(python3 --version 2>&1)"
echo "PORT: ${PORT:-8080}"
echo "streamlit: $(which streamlit 2>&1 || echo NOT FOUND)"
python3 -c "import streamlit; print(f'streamlit {streamlit.__version__}')"
python3 -c "import httpx; print(f'httpx {httpx.__version__}')"
python3 -c "import api.server; print('api.server OK')"
echo "=== Starting services ==="

echo "Starting API server on port 8000..."
uvicorn api.server:app --host 0.0.0.0 --port 8000 2>&1 &

echo "Starting Streamlit UI on port ${PORT:-8080}..."
exec streamlit run ui/app.py --server.port=${PORT:-8080} --server.address=0.0.0.0
