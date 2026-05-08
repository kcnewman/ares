FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

ENV UV_PYTHON=python3.12 \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock* ./
RUN mkdir -p core/pipeline api ui
RUN uv sync --frozen --no-dev

COPY core/ ./core/
COPY api/ ./api/
COPY ui/ ./ui/
COPY config/ ./config/
COPY .streamlit/ ./.streamlit/
COPY artifacts/ ./artifacts/

ENV BACKEND_URL="http://127.0.0.1:8000" \
    PORT=8080 \
    PATH="/app/.venv/bin:$PATH"

EXPOSE 8080

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
