import pytest
import pandas as pd
from unittest.mock import patch, mock_open, MagicMock
from ares.components.data_processing import DataProcessor


def test_string_cleaning_and_renaming(mock_process_config, dirty_df):
    with (
        patch("pandas.read_csv", return_value=dirty_df),
        patch("ares.utils.common.load_json", return_value={}),
    ):
        dp = DataProcessor(mock_process_config)
        cleaned = dp.clean_dataframe(dirty_df)

        assert "property_name" in cleaned.columns
        assert cleaned["property_name"].iloc[1] == "villa"
        assert "fetch_date" not in cleaned.columns


def test_outlier_removal_and_zero_price(mock_process_config):
    df = pd.DataFrame({"price": [0, -1, 100, 110, 120, 1000000]})
    with (
        patch("pandas.read_csv", return_value=df),
        patch("ares.utils.common.load_json", return_value={}),
    ):
        dp = DataProcessor(mock_process_config)
        trimmed = dp._trim_price_outliers(df)

        assert len(trimmed) == 3
        assert trimmed["price"].min() == 100


@patch("ares.components.data_processing.save_json")
@patch("ares.components.data_processing.maps.geocode")
def test_geocoding_logic(mock_geocode, mock_save, mock_process_config):
    with (
        patch("pandas.read_csv", return_value=pd.DataFrame()),
        patch(
            "ares.components.data_processing.load_json",
            return_value={"accra": {"lat": 5.6, "lng": -0.1}},
        ),
    ):
        dp = DataProcessor(mock_process_config)

        lat, lng = dp.get_lat_lng("Accra")
        assert lat == 5.6
        mock_geocode.assert_not_called()


def test_full_transform_execution(mock_process_config, dirty_df):
    """Checks if the main pipeline runs and produces the expected files."""
    with (
        patch("pandas.read_csv", return_value=dirty_df),
        patch("ares.utils.common.load_json", return_value={}),
        patch("pandas.DataFrame.to_csv") as mock_csv,
    ):
        dp = DataProcessor(mock_process_config)
        dp.transform()

        assert mock_csv.call_count == 2
        assert "preprocessed_train.csv" in mock_csv.call_args_list[0][0][0]
