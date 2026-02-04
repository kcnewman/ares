FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

ENV UV_PYTHON=python3.12
ENV UV_COMPILE_BYTECODE=1

COPY pyproject.toml uv.lock* ./

RUN uv sync --frozen --no-install-project --no-dev --no-group train --no-group ui

COPY src/ ./src/
COPY artifacts/ ./artifacts/
COPY config/ ./config/
COPY params.yaml schema.yaml app.py ./

ENV PYTHONPATH="/app/src:${PYTHONPATH}"
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "ares.api.main:app", "--host", "0.0.0.0", "--port", "8000"]