# Operations Runbook

This runbook covers day-to-day operation of ARES in local and cloud environments.

## 1. Produce Training Artifacts

Run the full pipeline:

```bash
uv run python main.py
```

Expected outputs include:

- `artifacts/model_trainer/model.joblib`
- `artifacts/feature_engineering/*.json`
- `artifacts/model_evaluation/metrics.json`
- `artifacts.zip`

## 2. Validate Artifacts Before Deployment

Minimum checks:

1. `artifacts/data_validation/status.txt` has `Validation status: True`
2. `artifacts/model_trainer/model.joblib` exists
3. `artifacts/feature_engineering/` contains locality stats JSON files
4. `artifacts/model_evaluation/metrics.json` exists

Optional smoke checks:

- API `/health`
- API `/predict` with sample payload
- Batch inference run against sample CSV

## 3. Local Container Build and Run

Build:

```bash
docker build -t ares:local .
```

Run:

```bash
docker run --rm -p 8080:8080 ares:local
```

Container behavior:

- FastAPI runs on internal port `8000`.
- Streamlit runs on exposed port `8080`.
- Streamlit calls backend using `BACKEND_URL=http://127.0.0.1:8000`.

## 4. CI/CD Deployment (Cloud Run)

Deployment workflow: `.github/workflows/deploy.yml`

Trigger:

- push to `main` with changes to `artifacts.zip`

Workflow summary:

1. Checkout repo
2. Unzip `artifacts.zip`
3. Authenticate to GCP using service account JSON
4. Build Docker image and push to Artifact Registry
5. Deploy image to Cloud Run service `ares-service`

Required GitHub secrets:

- `GCP_SA_KEY`
- `GCP_PROJECT_ID`

## 5. Observability

### Logs

- Local runtime logs: `logs/running_logs.log`
- Cloud Run logs: view in GCP logging console for `ares-service`

### Metrics

- Local evaluation metrics: `artifacts/model_evaluation/metrics.json`
- Experiment tracking: MLflow URI (`MLFLOW_TRACKING_URI` or default local SQLite URI)

## 6. Recovery and Rollback Notes

- If deployment fails, re-deploy the last known good image tag in Cloud Run.
- If inference quality drops, restore previous artifacts/model and redeploy.
- Keep versioned artifact snapshots for reliable rollback.

## 7. Security and Secrets

- Never commit `.env` or service account keys.
- Keep GCP credentials in GitHub Actions secrets only.
- Inject runtime config via environment variables, not source edits.
