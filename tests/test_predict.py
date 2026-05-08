"""Tests for lib/predict.py"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from lib.predict import predict, predict_from_dict


@pytest.fixture
def mock_model_and_metadata():
    with patch("lib.predict.load_model_and_metadata") as mock_load:
        mock_load.return_value = (
            _MockModel(),
            {
                "location_stats": {
                    "tesano": {
                        "median_log_price": 7.6,
                        "std_log_price": 0.3,
                        "iqr_log_price": 0.4,
                        "count": 50,
                    },
                },
                "categories": {
                    "house_type": ["apartment", "duplex"],
                    "condition": ["fairly used", "newly built"],
                    "furnishing": ["furnished", "unfurnished"],
                },
                "feature_columns": [
                    "bedrooms",
                    "bathrooms",
                    "24_hour_electricity",
                    "air_conditioning",
                    "wi_fi",
                    "total_amenities",
                    "luxury_count",
                    "loc_median_price",
                    "loc_volatility",
                    "loc_count",
                ],
                "global_stats": {
                    "median_log_price": 7.5,
                    "std_log_price": 0.5,
                    "iqr_log_price": 0.5,
                },
            },
        )
        yield mock_load


class _MockModel:
    """Mock CatBoost model returning constant predictions."""

    def predict(self, X):
        return np.full(len(X), 7.5)


def _make_row(**overrides):
    row = {
        "house_type": "apartment",
        "condition": "fairly used",
        "furnishing": "furnished",
        "loc": "tesano",
        "bedrooms": 2,
        "bathrooms": 1,
        "24_hour_electricity": 0,
        "air_conditioning": 0,
        "apartment": 0,
        "balcony": 0,
        "chandelier": 0,
        "dining_area": 0,
        "dishwasher": 0,
        "hot_water": 0,
        "kitchen_cabinets": 0,
        "kitchen_shelf": 0,
        "microwave": 0,
        "pop_ceiling": 0,
        "pre_paid_meter": 0,
        "refrigerator": 0,
        "tv": 0,
        "tiled_floor": 0,
        "wardrobe": 0,
        "wi_fi": 0,
    }
    row.update(overrides)
    return row


class TestPredict:
    def test_returns_dataframe(self, mock_model_and_metadata):
        df = pd.DataFrame([_make_row()])
        result = predict(df, model_dir=Path("models"))
        assert isinstance(result, pd.DataFrame)

    def test_contains_expected_columns(self, mock_model_and_metadata):
        df = pd.DataFrame([_make_row()])
        result = predict(df, model_dir=Path("models"))
        expected = {
            "estimated_price",
            "lower_band",
            "upper_band",
            "market_volatility_tier",
        }
        assert expected.issubset(result.columns)

    def test_estimated_price_positive(self, mock_model_and_metadata):
        df = pd.DataFrame([_make_row()])
        result = predict(df, model_dir=Path("models"))
        assert result["estimated_price"].iloc[0] > 0

    def test_lower_band_less_than_upper(self, mock_model_and_metadata):
        df = pd.DataFrame([_make_row()])
        result = predict(df, model_dir=Path("models"))
        assert result["lower_band"].iloc[0] < result["upper_band"].iloc[0]


class TestPredictFromDict:
    def test_returns_dict(self, mock_model_and_metadata):
        result = predict_from_dict(_make_row(), model_dir=Path("models"))
        assert isinstance(result, dict)

    def test_contains_all_keys(self, mock_model_and_metadata):
        result = predict_from_dict(_make_row(), model_dir=Path("models"))
        expected = {
            "estimated_price",
            "lower_band",
            "upper_band",
            "market_volatility_tier",
        }
        assert expected.issubset(result.keys())
