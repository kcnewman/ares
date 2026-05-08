#!/bin/bash
set -e

uvicorn api.app:app --host 0.0.0.0 --port 8000 &

for i in $(seq 1 30); do
    if curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1; then
        echo "API is ready"
        break
    fi
    echo "Waiting for API... attempt $i"
    sleep 1
done

exec streamlit run ui/app.py --server.port=${PORT:-8080} --server.address=0.0.0.0
