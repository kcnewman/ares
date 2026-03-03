import pytest
import numpy as np
from unittest.mock import patch
from box import ConfigBox
from ares.components.feature_engineering import EngineerFeatures
from ares.utils.volatility import shrink_to_global


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


def test_elite_features_are_computed(ef_instance, feature_df):
    """Ensures elite feature generation runs on edge values without crashing."""
    feature_df.loc[0, "price"] = 0

    prepared_df = ef_instance._add_amenity_features(feature_df)
    prepared_df = ef_instance._add_unit_density(prepared_df)
    prepared_df["loc_pi"] = 0.9
    prepared_df["class_pi"] = 0.8
    res = ef_instance._add_elite_features(prepared_df)

    assert "size_density_idx" in res.columns
    assert "class_luxury_premium" in res.columns
    assert np.isfinite(res["size_density_idx"]).all()


def test_bayesian_smoothing_logic(ef_instance):
    """Checks that low-support estimates shrink harder to global value."""
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
