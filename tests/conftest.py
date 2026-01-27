import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.train = "some_train.csv"
    config.test = "some_test.csv"
    config.geocode_cache = "some_cache.json"
    config.root_dir = "some_dir"
    return config


@pytest.fixture
def sample_data():
    import pandas as pd
    import numpy as np

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
