import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from joblib import dump
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from lib.features import (
    clean_dataframe,
    compute_location_stats,
    prepare_features,
    remove_price_outliers,
)
from lib.utils import logger, save_json

DEFAULT_DATA_PATH = Path("data/raw.csv")
DEFAULT_MODEL_DIR = Path("models")
CATBOOST_PARAMS = {
    "learning_rate": 0.06099,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "subsample": 0.8,
    "random_strength": 1.0,
    "min_data_in_leaf": 1,
    "random_seed": 42,
    "allow_writing_files": False,
    "verbose": 100,
}


def train(
    data_path: Path, model_dir: Path, test_size: float = 0.2, random_state: int = 2025
):
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows")

    df = clean_dataframe(df)
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0]
    logger.info(f"After cleaning: {len(df)} rows")

    df = remove_price_outliers(df, multiplier=1.5)

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    logger.info(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")

    location_stats = compute_location_stats(train_df)

    train_features, metadata = prepare_features(
        train_df, location_stats=location_stats, fit=True
    )
    test_features, _ = prepare_features(
        test_df, location_stats=location_stats, categories=metadata["categories"]
    )

    feature_cols = metadata["feature_columns"]
    X_train = train_features[feature_cols]
    y_train = train_features["log_price"]
    X_test = test_features[feature_cols]
    y_test = test_features["log_price"]

    logger.info(f"Feature matrix: {X_train.shape[1]} features, {X_train.shape[0]} rows")

    model = CatBoostRegressor(**CATBOOST_PARAMS)
    model.fit(X_train, y_train, eval_set=(X_test, y_test))

    y_pred = model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
    }
    logger.info(
        f"Test metrics: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}"
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.joblib"
    dump(model, model_path)
    logger.info(f"Model saved to {model_path}")

    global_log_prices = train_df["price"].apply(lambda x: np.log(max(x, 1)))
    metadata["location_stats"] = location_stats
    metadata["global_stats"] = {
        "median_log_price": float(np.median(global_log_prices)),
        "std_log_price": float(np.std(global_log_prices)),
        "iqr_log_price": float(
            np.percentile(global_log_prices, 75) - np.percentile(global_log_prices, 25)
        ),
    }
    metadata["metrics"] = metrics
    metadata["catboost_params"] = CATBOOST_PARAMS
    metadata["training_date"] = datetime.now().isoformat()
    metadata["n_train"] = len(train_df)
    metadata["n_test"] = len(test_df)

    metadata_path = model_dir / "metadata.json"
    save_json(metadata_path, metadata)
    logger.info(f"Metadata saved to {metadata_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--model-dir", type=str, default=str(DEFAULT_MODEL_DIR))
    args = parser.parse_args()

    train(Path(args.data), Path(args.model_dir))


if __name__ == "__main__":
    main()
