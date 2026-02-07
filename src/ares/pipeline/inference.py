"""
Inference pipeline for Housing Regression MLE.
- Aligns features strictly with training schema.
- Calculates IQR-based price bands.
"""

import pandas as pd
import numpy as np
from joblib import load
import argparse
from pathlib import Path

from ares.config.configuration import ConfigurationManager
from ares.components.feature_engineering import EngineerFeatures
from ares import logger

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = PROJECT_ROOT / "artifacts" / "model_trainer" / "model.joblib"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "inference" / "predictions.csv"

config_manager = ConfigurationManager()
fe_config = config_manager.get_feature_engineering_config()
fe_pipeline = EngineerFeatures(config=fe_config, mode="inference")


def predict(
    input_data: pd.DataFrame,
    model_path: Path | str = DEFAULT_MODEL,
    feature_pipeline: EngineerFeatures = fe_pipeline,
) -> pd.DataFrame:
    """Takes raw data, transforms it, aligns schema, and returns estimates with bands.

    Returns
        -------
        results: pd.DataFrame[str, float]
    """

    try:
        logger.info(f"Processing {len(input_data)} records for inference")

        transformed_data = feature_pipeline.run_pipeline(input_data)
        spread = transformed_data["loc_price_volatility"]

        training_features = [
            col for col in fe_pipeline.lists["required_columns"] if col != "log_price"
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
            },
            index=input_data.index,
        )

        currency_cols = ["estimated_price", "lower_band", "upper_band"]
        results[currency_cols] = results[currency_cols].round(-2)

        return results

    except Exception as e:
        logger.error(f"Inference pipeline failed: {str(e)}")
        raise e


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
