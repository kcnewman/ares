from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from core.pipeline.features import (
    _add_amenity_features,
    _add_elite_features,
    _add_unit_density,
    _haversine_distance,
    fit_features,
)
from core.volatility import shrink_to_global


@pytest.fixture
def fe_config():
    return {
        "feature_engineering": {
            "train": "train.csv",
            "test": "test.csv",
            "schema": "schema.json",
            "geocode_cache": "cache.json",
            "root_dir": Path("output"),
            "k_smoothing": 50,
            "elite_lux_threshold": 5,
            "elite_loc_pi_threshold": 0.8,
            "elite_areas_override": [],
        },
        "llm_service_url": "http://test-llm:8001",
    }


def test_math_logic_and_geo():
    dist = _haversine_distance(5.605, -0.166, 5.620, -0.173)
    assert 1.5 < dist < 2.5


def test_elite_features_are_computed(feature_schema, feature_df):

    luxury_cols = feature_schema["lists"]["amenities"]["luxury"]
    prepared_df = _add_amenity_features(feature_df, luxury_cols)
    prepared_df = _add_unit_density(prepared_df, feature_schema)
    prepared_df["loc_pi"] = 0.9
    prepared_df["class_pi"] = 0.8

    pipeline = {
        "loc_luxury_median": {"accra": 1.5, "east_legon": 1.0},
        "global_lux_median": 1.0,
    }
    res = _add_elite_features(prepared_df, pipeline)

    assert "size_density_idx" in res.columns
    assert "class_luxury_premium" in res.columns
    assert np.isfinite(res["size_density_idx"]).all()


def test_bayesian_smoothing_logic():
    global_std = 1.0
    local_std = 0.1

    smoothed_low = shrink_to_global(
        local_value=local_std,
        n_listings=1,
        global_value=global_std,
        k_smoothing=50,
    )
    smoothed_high = shrink_to_global(
        local_value=local_std,
        n_listings=500,
        global_value=global_std,
        k_smoothing=50,
    )

    assert abs(smoothed_low - global_std) < abs(local_std - global_std)
    assert abs(smoothed_high - local_std) < abs(smoothed_low - local_std)


def test_full_pipeline_persistence(fe_config, feature_schema, feature_df):
    with (
        patch("core.pipeline.features.load_json", side_effect=[feature_schema, {}]),
        patch("core.pipeline.features.pd.read_csv", return_value=feature_df),
        patch("core.pipeline.features.save_json") as mock_save,
        patch("pandas.DataFrame.to_csv") as mock_csv,
        patch("core.pipeline.features.create_directories"),
        patch("core.pipeline.features._fetch_location_tiers", return_value={"accra": {"tier": "established", "is_elite": False}, "east_legon": {"tier": "prime", "is_elite": True}}),
        patch("core.pipeline.features._fetch_amenity_tiers", return_value={"pool": "luxury", "wifi": "standard"}),
    ):
        fit_features(fe_config)

        assert mock_save.call_count >= 4
        assert mock_csv.call_count == 2
