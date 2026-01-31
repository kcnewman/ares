import pytest
import pandas as pd
from dataclasses import dataclass
from pathlib import Path


# ----- mock configs ----- #
@dataclass
class MockValConfig:
    data_dir: str = "dummy.csv"
    STATUS_FILE: str = "status.txt"
    all_schema: dict = None  # type: ignore


@dataclass
class MockSplitConfig:
    root_dir: str = "output"
    data_dir: str = "data.csv"


@dataclass
class MockProcessConfig:
    train: Path
    test: Path
    geocode_cache: Path
    root_dir: Path


@dataclass
class MockFEConfig:
    train: Path = Path("train.csv")
    test: Path = Path("test.csv")
    schema: Path = Path("schema.json")
    geocode_cache: Path = Path("cache.json")
    root_dir: Path = Path("output")


# ----- mock configs loading ----- #


@pytest.fixture
def mock_process_config():
    return MockProcessConfig(
        train=Path("train.csv"),
        test=Path("test.csv"),
        geocode_cache=Path("cache.json"),
        root_dir=Path("output"),
    )


@pytest.fixture
def mock_val_config():
    return MockValConfig(
        all_schema={"id": "int64", "locality": "object", "price": "float64"}
    )


@pytest.fixture
def mock_split_config():
    return MockSplitConfig()


@pytest.fixture
def mock_fe_config():
    return MockFEConfig()


# ----- mock data schema ----- #


@pytest.fixture
def feature_schema():
    return {
        "mappings": {
            "condition_transform": {"good": 2, "mint": 3},
            "furnishing_transform": {"none": 0, "full": 2},
            "location_class": {"accra": "tier1"},
            "property_density": {"apartment": 5},
        },
        "lists": {
            "amenities": {"luxury": ["pool", "wifi"]},
            "required_columns": [
                "id",
                "log_price",
                "loc_pi",
                "bath_per_bed",
                "luxury_score",
            ],
            "elite_areas": ["east_legon"],
        },
    }


@pytest.fixture
def feature_df():
    """Dataset designed to test Feature Engineering breakers (0 beds, etc)."""
    return pd.DataFrame(
        {
            "id": [1, 2],
            "loc": ["accra", "east_legon"],
            "price": [1000, 2000],
            "bedrooms": [2, 0],
            "bathrooms": [2, 1],
            "house_type": ["apartment", "apartment"],
            "condition": ["good", "mint"],
            "furnishing": ["none", "full"],
            "pool": [1, 0],
            "wifi": [1, 1],
            "lat": [5.60, 5.63],
            "lng": [-0.17, -0.15],
            "luxury_score": [2, 1],
            "loc_pi": [0.9, 0.5],
        }
    )


@pytest.fixture
def valid_df():
    return pd.DataFrame(
        {
            "id": range(60),
            "locality": ["Urban"] * 55 + ["Rural"] * 5,
            "price": [200000.0] * 60,
        }
    )


@pytest.fixture
def dirty_df():
    return pd.DataFrame(
        {
            "Property-Name": [" Villa!", "Villa!", "Hut"],
            "price": [100.0, 100.0, 10000000.0],
            "url": ["u1", "u1", "u3"],
            "location": ["Accra", "Accra", "Legon"],
            "fetch_date": ["2023-01-01"] * 3,
        }
    )
