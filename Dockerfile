FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install system dependencies required for building C extensions (like Catboost/XGBoost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

ENV UV_PYTHON=python3.12 \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock* README.md ./

COPY src/ ./src/

RUN uv sync --frozen --no-dev

# Copy the rest of the application code
COPY artifacts/ ./artifacts/
COPY config/ ./config/
COPY pages/ ./pages/
COPY params.yaml schema.yaml app.py utils.py ./
COPY .streamlit/ ./.streamlit/

ENV BACKEND_URL="http://127.0.0.1:8000" \
    PORT=8080 \
    PYTHONPATH="/app/src" \
    PATH="/app/.venv/bin:$PATH"

# Expose the Streamlit port
EXPOSE 8080

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
