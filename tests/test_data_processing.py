from unittest.mock import patch

import pandas as pd

from core.pipeline.data import (
    _add_lat_lng,
    _clean_dataframe,
    _filter_price_outliers,
    process,
)


def test_string_cleaning_and_renaming(dirty_df):
    with (
        patch("core.pipeline.data.load_json", return_value={}),
        patch("core.pipeline.data._build_maps_client", return_value=None),
    ):
        cleaned = _clean_dataframe(dirty_df)

        assert "property_name" in cleaned.columns
        assert cleaned["property_name"].iloc[1] == "villa"
        assert "fetch_date" not in cleaned.columns


def test_outlier_removal_and_zero_price():
    df = pd.DataFrame({"price": [0, -1, 100, 110, 120, 1000000]})
    trimmed = _filter_price_outliers(df)

    assert len(trimmed) == 3
    assert trimmed["price"].min() == 100


@patch("core.pipeline.data.save_json")
def test_geocoding_logic(mock_save):
    df = pd.DataFrame({"loc": ["Accra", "accra", "Legon"]})
    cache = {"accra": {"lat": 5.6, "lng": -0.1}}

    result = _add_lat_lng(
        df, maps_client=None, geocode_cache=cache, cache_path="cache.json"
    )

    assert result["lat"].iloc[0] == 5.6
    assert result["lat"].iloc[1] == 5.6
    mock_save.assert_not_called()


def test_full_transform_execution(dirty_df):
    config = {
        "data_processing": {
            "train": "train.csv",
            "test": "test.csv",
            "geocode_cache": "cache.json",
            "root_dir": "output",
        }
    }
    with (
        patch("core.pipeline.data.pd.read_csv", return_value=dirty_df),
        patch("core.pipeline.data.load_json", return_value={}),
        patch("core.pipeline.data._build_maps_client", return_value=None),
        patch("core.pipeline.data.create_directories"),
        patch("pandas.DataFrame.to_csv") as mock_csv,
    ):
        process(config)

        assert mock_csv.call_count == 2
        assert "preprocessed_train.csv" in mock_csv.call_args_list[0][0][0]
