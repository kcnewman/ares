# Configuration Reference

ARES behavior is controlled through three files and a small set of environment variables.

## Config Files

### `config/config.yaml`

Defines stage-level wiring and artifact locations.

Key sections:

- `artifacts_root`: top-level artifact directory.
- `data_validation`: raw data path and validation status path.
- `data_split`: split stage inputs and outputs.
- `data_processing`: train/eval inputs and geocode cache path.
- `feature_engineering`: processed inputs and schema/cache files.
- `model_trainer`: feature files, output model path.
- `model_evaluation`: test feature file, model path, metric output.

### `params.yaml`

Contains model hyperparameters under `CatBoost`.

Example parameters:

- `learning_rate`
- `depth`
- `l2_leaf_reg`
- `subsample`
- `random_seed`

### `schema.yaml`

Defines raw input schema and target information.

- `COLUMNS`: expected raw columns and dtypes for validation.
- `TARGET_COLUMN.name`: model training target (`log_price`).

## Environment Variables

- `GOOGLE_MAPS_KEY`: optional, used when locality coordinates are not already cached.
- `MLFLOW_TRACKING_URI`: optional, overrides default `sqlite:///experiments/mlflow.db`.
- `BACKEND_URL`: used by Streamlit frontend to reach FastAPI backend.
- `PORT`: containerized Streamlit port (defaults to `8080` in deployment).

## Configuration Flow

`ConfigurationManager` (`src/ares/config/configuration.py`) does the following:

1. Loads `config/config.yaml`, `params.yaml`, `schema.yaml`.
2. Creates stage directories as needed.
3. Builds typed config entities for each pipeline stage.
4. Resolves MLflow URI from env var or default.

## Changing Artifact Locations

If you move artifact paths:

1. Update `config/config.yaml`.
2. Ensure consuming stages still point to the new location.
3. Run the pipeline end-to-end once to verify handoffs.
4. Update docs if paths changed.

## Notes on Raw Data Contract

- Validation is strict on columns and dtypes.
- Unexpected columns fail validation.
- Missing columns fail validation.
- Validation result is persisted to `artifacts/data_validation/status.txt`.
