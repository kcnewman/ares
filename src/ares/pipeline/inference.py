"""
Inference pipeline for Housing Regression MLE.
- Aligns features strictly with training schema.
- Calculates IQR-based price bands.
"""

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

from ares import logger
from ares.components.feature_engineering import EngineerFeatures
from ares.config.configuration import ConfigurationManager
from ares.utils.volatility import (
    classify_volatility_tier,
    derive_volatility_thresholds,
    log_iqr_to_relative_pct,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = PROJECT_ROOT / "artifacts" / "model_trainer" / "model.joblib"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "inference" / "predictions.csv"


@lru_cache(maxsize=1)
def _get_feature_pipeline() -> EngineerFeatures:
    config_manager = ConfigurationManager()
    fe_config = config_manager.get_feature_engineering_config()
    return EngineerFeatures(config=fe_config, mode="inference")


def _load_volatility_thresholds(
    feature_pipeline: EngineerFeatures, spread: pd.Series
) -> tuple[float, float]:
    q25 = float(
        feature_pipeline.global_ref.get(
            "volatility_q25",
            feature_pipeline.global_ref.get("vol_q25", np.nan),
        )
    )
    q75 = float(
        feature_pipeline.global_ref.get(
            "volatility_q75",
            feature_pipeline.global_ref.get("vol_q75", np.nan),
        )
    )
    if np.isfinite(q25) and np.isfinite(q75) and q75 > q25:
        return q25, q75

    loc_iqr_values = pd.Series(list(feature_pipeline.loc_iqr.values()), dtype=float)
    loc_iqr_values = loc_iqr_values[np.isfinite(loc_iqr_values)]
    if loc_iqr_values.empty:
        return derive_volatility_thresholds(spread)
    return derive_volatility_thresholds(loc_iqr_values)


def predict(
    input_data: pd.DataFrame,
    model_path: Path | str = DEFAULT_MODEL,
    feature_pipeline: EngineerFeatures | None = None,
) -> pd.DataFrame:
    """Takes raw data, transforms it, aligns schema, and returns estimates with bands.

    Returns
        -------
        results: pd.DataFrame[str, float]
    """

    try:
        feature_pipeline = feature_pipeline or _get_feature_pipeline()
        logger.info(f"Processing {len(input_data)} records for inference")

        transformed_data = feature_pipeline.run_pipeline(input_data)
        spread = transformed_data["loc_price_volatility"].astype(float)
        vol_q25, vol_q75 = _load_volatility_thresholds(feature_pipeline, spread)
        volatility_tier = spread.apply(
            lambda value: classify_volatility_tier(float(value), vol_q25, vol_q75)
        )
        volatility_pct = spread.apply(log_iqr_to_relative_pct)

        training_features = [
            col
            for col in feature_pipeline.lists["required_columns"]
            if col != "log_price"
        ]
        model_input = transformed_data[training_features]

        model = load(model_path)
        logger.info("Executing CatBoost inference")
        log_preds = model.predict(model_input)

        lower_log = log_preds - (spread / 2)
        upper_log = log_preds + (spread / 2)

        results = pd.DataFrame(
            {
                "estimated_price": np.exp(log_preds),
                "lower_band": np.exp(lower_log),
                "upper_band": np.exp(upper_log),
                "market_volatility_idx": spread,
                "market_volatility_pct": volatility_pct,
                "market_volatility_tier": volatility_tier,
            },
            index=input_data.index,
        )

        currency_cols = ["estimated_price", "lower_band", "upper_band"]
        results[currency_cols] = results[currency_cols].round(-2)

        return results

    except Exception as e:
        logger.error(f"Inference pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARES Inference Pipeline")
    parser.add_argument(
        "--input", type=str, required=True, help="Path to raw CSV input"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL),
        help="Path to .joblib model file",
    )

    args = parser.parse_args()

    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(args.input)
    preds_df = predict(raw_df, args.model)

    final_output = pd.concat([raw_df, preds_df], axis=1)
    display_cols = [
        "loc",
        "house_type",
        "bedrooms",
        "lower_band",
        "estimated_price",
        "upper_band",
    ]

    print("\n--- Ares Prediction Results ---")
    print(final_output[display_cols].head())

    final_output.to_csv(DEFAULT_OUTPUT, index=False)
    logger.info(f"Results saved to {DEFAULT_OUTPUT}")
