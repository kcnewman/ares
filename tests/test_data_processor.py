import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from ares.components.data_processing import DataProcessor


@pytest.fixture
def mock_config():
    """Mocks the configuration object."""
    config = MagicMock()
    config.train = "some_train.csv"
    config.test = "some_test.csv"
    config.geocode_cache = "some_cache.json"
    config.root_dir = "some_dir"
    return config


@pytest.fixture
def sample_data():
    """A messy dataframe to test cleaning and pipeline flow."""
    return pd.DataFrame(
        {
            "Property-Name": [
                "House",
                "House",
                "Normal",
                "Normal",
                "Outlier",
                "Missing",
            ],
            "price": [100, 100, 110, 120, 100000000, 150],
            "url": ["url1", "url1", "url2", "url3", "url4", "url5"],
            "location": ["Accra", "Accra", "Accra", "Accra", "Kumasi", np.nan],
        }
    )


@patch("pandas.read_csv")
@patch("ares.components.data_processing.load_json", return_value={})
def test_full_transform_pipeline(mock_load, mock_read, mock_config, sample_data):
    """Tests the entire flow from start to finish."""
    mock_read.return_value = sample_data.copy()

    with patch("pandas.DataFrame.to_csv") as mock_to_csv:
        dp = DataProcessor(mock_config)
        dp.transform()

        assert len(dp.train) == 3
        assert "property_name" in dp.train.columns
        assert mock_to_csv.call_count == 2


def test_clean_strings_logic(mock_config):
    """Tests if strings are properly lowercase and stripped of special chars."""
    df = pd.DataFrame({"name": ["  Luxury-Villa 123!  "]})
    with patch("pandas.read_csv", return_value=df):
        dp = DataProcessor(mock_config)
        cleaned = dp._clean_strings(df.copy())
        assert cleaned.iloc[0]["name"] == "luxury villa"


@patch("ares.components.data_processing.maps.geocode")
@patch("ares.components.data_processing.save_json")
def test_get_lat_lng_with_cache(mock_save, mock_geocode, mock_config):
    """Tests that cache is checked and API is called if missing."""
    with patch("pandas.read_csv", return_value=pd.DataFrame()):
        dp = DataProcessor(mock_config)

        dp.geocode_cache = {"accra": {"lat": 5.6, "lng": -0.1}}
        lat, lng = dp.get_lat_lng("Accra")
        assert lat == 5.6
        mock_geocode.assert_not_called()

        mock_geocode.return_value = [
            {"geometry": {"location": {"lat": 6.0, "lng": 0.0}}}
        ]
        lat, lng = dp.get_lat_lng("Tema")
        assert lat == 6.0
        assert mock_geocode.called


def test_trim_outliers_calculation(mock_config):
    """Tests that the IQR logic actually removes high-end values."""
    df = pd.DataFrame({"price": [10, 11, 12, 13, 1000]})
    with patch("pandas.read_csv", return_value=df):
        dp = DataProcessor(mock_config)
        trimmed = dp._trim_price_outliers(df)
        assert 1000 not in trimmed["price"].values
        assert len(trimmed) == 4


def test_handle_zero_price(mock_config):
    """Checks if the outlier logic crashes on zero/negative prices."""
    df = pd.DataFrame({"price": [0, -10, 100, 110, 120]})
    with patch("pandas.read_csv", return_value=df):
        dp = DataProcessor(mock_config)
        trimmed = dp._trim_price_outliers(df)
        assert len(trimmed) > 0


@patch("ares.components.data_processing.maps.geocode")
def test_get_lat_lng_api_error(mock_geocode, mock_config):
    """Tests that the system survives a Google API failure."""
    with patch("pandas.read_csv", return_value=pd.DataFrame()):
        dp = DataProcessor(mock_config)
        mock_geocode.side_effect = Exception("API Down")

        lat, lng = dp.get_lat_lng("Accra")
        assert lat is None
        assert lng is None
