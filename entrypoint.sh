#!/bin/bash
set -e

# Bind to 0.0.0.0 so it's accessible within the container network
# We use a port (8000) different from the Streamlit port (8080)
uvicorn ares.api.main:app --host 0.0.0.0 --port 8000 &

# Wait for the backend to initialize
sleep 3

# Start Streamlit on the port defined by the Docker ENV (default 8080)
streamlit run app.py --server.port=${PORT:-8080} --server.address=0.0.0.0