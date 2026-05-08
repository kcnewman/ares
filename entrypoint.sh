#!/bin/bash
set -e

cleanup() {
    echo "Shutting down..."
    kill $API_PID 2>/dev/null || true
    wait $API_PID 2>/dev/null || true
}
trap cleanup SIGTERM SIGINT EXIT

echo "Starting API server..."
uvicorn api.server:app --host 0.0.0.0 --port 8000 &
API_PID=$!

for i in $(seq 1 30); do
    if curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1; then
        echo "API is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "API failed to start within 30 seconds"
        exit 1
    fi
    echo "Waiting for API... attempt $i"
    sleep 1
done

echo "Starting Streamlit UI..."
exec streamlit run ui/app.py --server.port=${PORT:-8080} --server.address=0.0.0.0
