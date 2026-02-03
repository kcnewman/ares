# Ares Project

ARES (Automated Residential Estimation System) is a machine learning pipeline designed to estimate rental price ranges for residential properties in the Greater Accra Region, Ghana. The project follows MLOps best practices with modular pipelines, experiment tracking via MLflow, containerization, cloud deployment, and comprehensive testing. The system includes both a REST API and a Streamlit dashboard for interactive model inference.

---

## Core Stack

- **Python**: 3.12
- **Model**: CatBoostRegressor
- **API**: FastAPI
- **Experiment Tracking**: MLflow
- **Dependency Management**: uv

---

## Full Pipeline

```
Raw Data (CSV)
   ↓
Data Validation
   ↓
Stratified Split (by locality)
   ↓
Data Processing + Geocoding
   ↓
Feature Engineering
   ↓
Model Training
   ↓
Evaluation + MLflow Logging
   ↓
Shared Artifacts → Inference (Batch + API)
```

---

## Configuration System

Pipeline behavior is fully controlled via YAML:

- `config/config.yaml` — artifact paths and pipeline wiring
- `config/params.yaml` — model hyperparameters
- `config/schema.yaml` — input schema and target definition

### ConfigurationManager

- Loads all configuration files
- Creates required artifact directories
- Injects target column (`log_price`)
- Resolves MLflow tracking URI (supports env override)

No stage hardcodes paths or parameters.

---

## Artifact Layout

```
artifacts/
├── cache/                    # reusable lookup data
├── data/                     # raw input
├── data_split/               # train / eval CSVs
├── data_processing/          # cleaned data
├── feature_engineering/      # features + locality stats
├── model_trainer/            # trained model
├── model_evaluation/         # metrics
└── inference/                # batch predictions
```

---

## Running Training

### Prerequisites

- Python 3.12
- Dependencies installed via `uv`
- Raw CSV available at the path defined in `config.yaml`

### Optional Environment Variables

```bash
export GOOGLE_MAPS_KEY=...
export MLFLOW_TRACKING_URI=...
```

### Execute

```bash
python main.py
```

If validation fails, downstream stages abort.

---

## Pipeline Stages

### Data Validation

- Enforces schema from `schema.yaml`
- Rejects unexpected columns
- Writes status to `artifacts/data_validation/status.txt`

### Data Splitting

- Groups rare localities into `OTHER`
- Stratified split by locality

### Data Processing

- Cleans and normalizes inputs
- Handles missing values and outliers
- Geocodes localities (cached)

### Feature Engineering

- Generates model-ready features
- Computes and persists locality-level statistics:
  - Price indices
  - Volatility (IQR, std)
  - Amenity-based luxury indices
  - Geographic metrics

### Model Training

- Trains CatBoostRegressor
- Target: `log_price`
- Saves trained model
- Computes MAE, RMSE, R²

### Model Evaluation

- Reloads model and features
- Recomputes metrics
- Logs params, metrics, and model to MLflow
- Persists metrics as JSON

---

## Uncertainty Bands

Predictions are made in **log-price space**.
Locality-level volatility defines uncertainty:

```
log_price ± 0.5 × locality_volatility
```

Results are exponentiated back to price space.
This yields interpretable, locality-aware bounds.

---

## Inference

### Batch Inference (CLI)

```bash
python src/ares/pipeline/inference.py \
  --input path/to/raw.csv \
  --model artifacts/model_trainer/model.joblib
```

Outputs are written to:

```
artifacts/inference/predictions.csv
```

### Online Inference (FastAPI)

- App: `src/ares/api/main.py`
- Endpoints:
  - `GET /`
  - `GET /health`
  - `POST /predict`

`/predict` returns:

- `estimated_price`
- `lower_band`
- `upper_band`
- `market_volatility_idx`

Errors surface as HTTP 500 with logging.

Run locally:

```bash
uvicorn src.ares.api.main:app --reload
```

---

## MLflow Integration

Default tracking URI:

```
sqlite:///experiments/mlflow.db
```

Logged:

- Hyperparameters
- Evaluation metrics
- Model artifact

Supports later registration and reuse.

---

## Testing

- Pytest-based tests under `tests/`
- Covers:
  - Schema validation
  - Splitting logic
  - Feature consistency
  - Edge cases (rare localities, missing geo data)

Run:

```bash
pytest
```

---

## Troubleshooting

Validation failed?

```
artifacts/data_validation/status.txt
```

Unexpected performance?

```
artifacts/model_evaluation/metrics.json
```

Weird predictions?

```
artifacts/inference/predictions.csv
```

---
