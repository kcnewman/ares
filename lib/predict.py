from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

from lib.features import prepare_features
from lib.utils import load_json

DEFAULT_MODEL_DIR = Path("models")


@lru_cache(maxsize=1)
def load_model_and_metadata(model_dir: Path = DEFAULT_MODEL_DIR):
    model_path = model_dir / "model.joblib"
    metadata_path = model_dir / "metadata.json"

    model = load(model_path)
    metadata = load_json(metadata_path)
    return model, metadata


def predict(
    input_df: pd.DataFrame, model_dir: Path = DEFAULT_MODEL_DIR
) -> pd.DataFrame:
    model, metadata = load_model_and_metadata(model_dir)

    location_stats = metadata.get("location_stats", {})
    categories = metadata.get("categories", {})
    feature_columns = metadata.get("feature_columns", [])

    features, _ = prepare_features(
        input_df,
        location_stats=location_stats,
        categories=categories,
        fit=False,
    )

    for col in feature_columns:
        if col not in features.columns:
            features[col] = 0

    X = features[feature_columns]
    log_preds = model.predict(X)

    global_iqr = metadata.get("global_stats", {}).get("iqr_log_price", 0.5)
    loc_iqr_values = (
        features["loc_volatility"].values
        if "loc_volatility" in features.columns
        else np.full(len(log_preds), global_iqr)
    )
    loc_volatility = np.where(loc_iqr_values > 0, loc_iqr_values, global_iqr)

    n_listings = (
        features["loc_count"].values
        if "loc_count" in features.columns
        else np.full(len(log_preds), 0)
    )
    spread = loc_volatility * (1 + np.exp(-np.clip(n_listings / 10, 0, 5)))

    estimated_price = np.exp(log_preds)
    lower_band = np.exp(log_preds - spread / 2)
    upper_band = np.exp(log_preds + spread / 2)

    all_iqrs = np.where(
        loc_iqr_values > 0, loc_iqr_values, np.full(len(log_preds), global_iqr)
    )
    global_q25 = metadata.get("global_stats", {}).get("iqr_log_price", global_iqr) * 0.5
    global_q75 = metadata.get("global_stats", {}).get("iqr_log_price", global_iqr) * 1.5

    def classify_tier(val):
        if val <= global_q25:
            return "Stable"
        elif val <= global_q75:
            return "Moderate"
        return "Volatile"

    results = pd.DataFrame(
        {
            "estimated_price": estimated_price,
            "lower_band": lower_band,
            "upper_band": upper_band,
            "market_volatility_idx": all_iqrs,
            "market_volatility_pct": (np.exp(all_iqrs) - 1) * 100,
            "market_volatility_tier": [classify_tier(v) for v in all_iqrs],
        }
    )

    return results


def predict_from_dict(features: dict, model_dir: Path = DEFAULT_MODEL_DIR) -> dict:
    df = pd.DataFrame([features])
    result = predict(df, model_dir)
    row = result.iloc[0]
    return {
        "estimated_price": round(float(row["estimated_price"]), 2),
        "lower_band": round(float(row["lower_band"]), 2),
        "upper_band": round(float(row["upper_band"]), 2),
        "market_volatility_idx": round(float(row["market_volatility_idx"]), 4),
        "market_volatility_pct": round(float(row["market_volatility_pct"]), 2),
        "market_volatility_tier": str(row["market_volatility_tier"]),
    }
