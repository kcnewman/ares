"""Tests for lib/features.py"""

import numpy as np
import pandas as pd
import pytest

from lib.features import (
    clean_dataframe,
    compute_location_stats,
    prepare_features,
    remove_price_outliers,
)


@pytest.fixture
def raw_df():
    return pd.DataFrame(
        {
            "house_type": ["Apartment", "Duplex", "Apartment"],
            "Condition": ["Fairly Used", "Newly Built", "Fairly Used"],
            "Furnishing": ["Furnished", "Unfurnished", "Furnished"],
            "bedrooms": [2, 3, 2],
            "bathrooms": [1, 2, 1],
            "price": [2000, 5000, 2200],
            "loc": ["tesano", "oshodu", "tesano"],
            "24-hour Electricity": [1, 0, 1],
            "Air Conditioning": [1, 1, 0],
            "Wi-Fi": [0, 1, 1],
            "Apartment": [0, 0, 0],
            "Balcony": [0, 0, 0],
            "Chandelier": [0, 0, 0],
            "Dining Area": [0, 0, 0],
            "Dishwasher": [0, 0, 0],
            "Hot Water": [0, 0, 0],
            "Kitchen Cabinets": [0, 0, 0],
            "Kitchen Shelf": [0, 0, 0],
            "Microwave": [0, 0, 0],
            "Pop Ceiling": [0, 0, 0],
            "Pre-Paid Meter": [0, 0, 0],
            "Refrigerator": [0, 0, 0],
            "TV": [0, 0, 0],
            "Tiled Floor": [0, 0, 0],
            "Wardrobe": [0, 0, 0],
            "url": ["http://a.com", "http://b.com", "http://c.com"],
            "fetch_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "Property Size": [100, 200, 150],
        }
    )


class TestCleanDataframe:
    def test_renames_columns(self, raw_df):
        result = clean_dataframe(raw_df)
        assert "condition" in result.columns
        assert "furnishing" in result.columns
        assert "24_hour_electricity" in result.columns
        assert "wi_fi" in result.columns

    def test_lowercases_strings(self, raw_df):
        result = clean_dataframe(raw_df)
        assert result["condition"].iloc[0] == "fairly used"
        assert result["furnishing"].iloc[0] == "furnished"

    def test_amenities_are_integers(self, raw_df):
        result = clean_dataframe(raw_df)
        assert result["24_hour_electricity"].dtype == int
        assert result["wi_fi"].dtype == int

    def test_drops_locality_if_present(self, raw_df):
        df = raw_df.copy()
        df["locality"] = ["a", "b", "c"]
        result = clean_dataframe(df)
        assert "locality" not in result.columns


class TestRemovePriceOutliers:
    def test_removes_extreme_prices(self):
        df = pd.DataFrame({"price": [1000, 1200, 1100, 100000]})
        result = remove_price_outliers(df, multiplier=1.5)
        assert len(result) == 3

    def test_keeps_all_when_no_outliers(self):
        df = pd.DataFrame({"price": [1000, 1100, 1200, 1050]})
        result = remove_price_outliers(df, multiplier=1.5)
        assert len(result) == 4


class TestComputeLocationStats:
    def test_returns_stats_per_location(self):
        df = pd.DataFrame(
            {
                "loc": ["a", "a", "b"],
                "price": [1000, 2000, 3000],
            }
        )
        stats = compute_location_stats(df)
        assert "a" in stats
        assert "b" in stats
        assert stats["a"]["count"] == 2
        assert stats["b"]["count"] == 1

    def test_returns_empty_for_no_loc_column(self):
        df = pd.DataFrame({"price": [1000, 2000]})
        stats = compute_location_stats(df)
        assert stats == {}


class TestPrepareFeatures:
    def test_adds_log_price(self, raw_df):
        df = clean_dataframe(raw_df)
        result, metadata = prepare_features(df, fit=True)
        assert "log_price" in result.columns
        assert np.allclose(result["log_price"], np.log(df["price"]))

    def test_adds_total_amenities(self, raw_df):
        df = clean_dataframe(raw_df)
        result, _ = prepare_features(df, fit=True)
        assert "total_amenities" in result.columns
        assert result["total_amenities"].tolist() == [2, 2, 2]

    def test_adds_luxury_count(self, raw_df):
        df = clean_dataframe(raw_df)
        result, _ = prepare_features(df, fit=True)
        assert "luxury_count" in result.columns
        assert result["luxury_count"].tolist() == [1, 2, 1]

    def test_returns_feature_columns_in_metadata(self, raw_df):
        df = clean_dataframe(raw_df)
        _, metadata = prepare_features(df, fit=True)
        assert "feature_columns" in metadata
        assert len(metadata["feature_columns"]) > 0

    def test_fills_missing_features_with_zero(self, raw_df):
        df = clean_dataframe(raw_df)
        result, metadata = prepare_features(df, fit=True)
        for col in metadata["feature_columns"]:
            assert col in result.columns
