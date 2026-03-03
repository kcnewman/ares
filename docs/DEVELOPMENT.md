# Development Guide

This guide covers local development workflow for ARES.

## Prerequisites

- Python `3.12.x`
- `uv`
- Dataset at `artifacts/data/raw.csv` (or update `config/config.yaml`)

## Local Environment

```bash
uv sync --dev
```

Optional environment variables:

```bash
export GOOGLE_MAPS_KEY="your-key"
export MLFLOW_TRACKING_URI="sqlite:///experiments/mlflow.db"
export BACKEND_URL="http://127.0.0.1:8000"
```

## Running the Full Pipeline

```bash
uv run python main.py
```

This runs all stages and then creates `artifacts.zip`.

## Running Individual Stages

Use module entrypoints when developing a single stage:

```bash
uv run python -m ares.pipeline.data_validation
uv run python -m ares.pipeline.data_split
uv run python -m ares.pipeline.data_processing
uv run python -m ares.pipeline.feature_engineering
uv run python -m ares.pipeline.model_trainer
uv run python -m ares.pipeline.model_evaluation
```

## Running API and UI Locally

Start the API:

```bash
uv run uvicorn ares.api.main:app --host 127.0.0.1 --port 8000 --reload
```

Start Streamlit in another terminal:

```bash
BACKEND_URL="http://127.0.0.1:8000" uv run streamlit run app.py
```

## Running Batch Inference

```bash
uv run python src/ares/pipeline/inference.py \
  --input artifacts/data/raw.csv \
  --model artifacts/model_trainer/model.joblib
```

## Tests

```bash
uv run pytest -q
```

Current tests focus on:

- schema validation behavior
- split gating and stratification
- data cleaning and geocoding cache behavior
- feature engineering edge cases

## Logs and Artifacts During Development

- Runtime log: `logs/running_logs.log`
- Validation gate file: `artifacts/data_validation/status.txt`
- Evaluation metrics: `artifacts/model_evaluation/metrics.json`
- Batch inference output: `artifacts/inference/predictions.csv`

## Development Practices

- Keep stage logic deterministic and artifact-driven.
- Preserve training-serving feature parity when changing feature engineering.
- Update docs and tests together for contract changes.
