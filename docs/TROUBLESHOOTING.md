# Troubleshooting

Common issues and how to diagnose them.

## Validation Fails at Pipeline Start

Symptoms:

- Pipeline stops before split or downstream stages.
- `artifacts/data_validation/status.txt` contains failure details.

Checks:

1. Confirm raw input path in `config/config.yaml` exists.
2. Compare raw CSV columns against `schema.yaml`.
3. Verify raw CSV dtypes match the declared schema.

## Data Split Does Not Run

Symptoms:

- No `artifacts/data_split/train.csv` or `eval.csv`.

Checks:

1. Ensure `artifacts/data_validation/status.txt` exists.
2. Ensure first line is `Validation status: True`.

## Geocoding Errors in Data Processing

Symptoms:

- Log entries like `Error geocoding ...`.

Checks:

1. Set a valid `GOOGLE_MAPS_KEY` in environment.
2. Confirm network access in your runtime environment.
3. Verify `artifacts/cache/geocode_cache.json` is readable/writable.

## Streamlit UI Shows "Schema file missing"

Symptoms:

- Frontend displays missing schema error.

Checks:

1. Confirm `artifacts/cache/schema.json` exists.
2. Re-run feature engineering or full pipeline to regenerate cache files.

## API Returns 500 on `/predict`

Symptoms:

- Response body: `Internal Server Error during inference.`

Checks:

1. Confirm `artifacts/model_trainer/model.joblib` exists.
2. Confirm feature stats files exist in `artifacts/feature_engineering/`.
3. Verify request payload uses expected field names and all required keys.
4. Inspect `logs/running_logs.log` for stack trace details.

## MLflow Logging Fails

Symptoms:

- Evaluation stage errors when writing run data.

Checks:

1. Verify `MLFLOW_TRACKING_URI` is valid.
2. If using local SQLite default, ensure `experiments/` is writable.
3. Confirm `mlflow` dependency is installed (`uv sync --dev`).

## Docker Build or Runtime Issues

Symptoms:

- Image build fails or container starts without working app.

Checks:

1. Ensure `artifacts/` is populated before building image.
2. Confirm `entrypoint.sh` is executable.
3. Verify container exposes `PORT=8080` for Streamlit on Cloud Run.

## Quick Health Commands

```bash
# API health
curl http://127.0.0.1:8000/health

# Local tests
uv run pytest -q

# End-to-end pipeline
uv run python main.py
```
