import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from box import ConfigBox
from ares.components.feature_engineering import EngineerFeatures


@pytest.fixture
def ef_instance(mock_fe_config, feature_schema, feature_df):
    """Utility to create a FE instance with mocked loads."""
    with (
        patch(
            "ares.components.feature_engineering.load_json",
            side_effect=[ConfigBox(feature_schema), {}],
        ),
        patch("pandas.read_csv", return_value=feature_df),
    ):
        return EngineerFeatures(mock_fe_config)


def test_math_logic_and_geo(ef_instance):
    dist = ef_instance._haversine_distance(5.605, -0.166, 5.620, -0.173)
    assert 1.5 < dist < 2.5


def test_zero_division(ef_instance, feature_df):
    """Ensures 0 bedrooms or 0 price doesn't crash the pipeline."""
    feature_df.loc[0, "price"] = 0

    prepared_df = ef_instance._add_amenity_features(feature_df)
    prepared_df["loc_pi"] = 0.9
    res = ef_instance._add_elite_features(prepared_df)

    assert res.loc[1, "bath_per_bed"] == 0
    assert np.isfinite(res["bath_per_bed"]).all()


def test_bayesian_smoothing_logic(ef_instance):
    """Checks if small samples are pulled toward the global median."""
    ef_instance.global_ref = {"std": 1.0}

    row_low = {"loc": "rare", "n_listings": 1, "std_log": 0.1}
    stats_low = ef_instance._compute_bayesian_stats(row_low, K=50)

    row_high = {"loc": "busy", "n_listings": 500, "std_log": 0.1}
    stats_high = ef_instance._compute_bayesian_stats(row_high, K=50)

    assert stats_low["loc_trust_score"] < stats_high["loc_trust_score"]
    assert stats_low["loc_std_dev"] > 0.5


def test_full_pipeline_persistence(mock_fe_config, feature_schema, feature_df):
    """Verifies the transform() method writes all required artifacts."""
    with (
        patch(
            "ares.components.feature_engineering.load_json",
            side_effect=[ConfigBox(feature_schema), {}],
        ),
        patch("pandas.read_csv", return_value=feature_df),
        patch("ares.components.feature_engineering.save_json") as mock_save,
        patch("pandas.DataFrame.to_csv") as mock_csv,
    ):
        ef = EngineerFeatures(mock_fe_config)
        ef.transform()

        assert mock_save.call_count >= 4
        assert mock_csv.call_count == 2
