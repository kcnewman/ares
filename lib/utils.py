import json
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


def setup_logger(name="ares"):
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger(name)
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, level_name, logging.INFO))

    if logger.handlers:
        return logger

    file_handler = RotatingFileHandler(
        log_dir / "running_logs.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
    )
    file_handler.setFormatter(
        logging.Formatter("[%(asctime)s: %(levelname)s: %(message)s]")
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


logger = setup_logger()


def save_json(path: Path, data: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


RAW_COLUMN_MAP = {
    "Condition": "condition",
    "Furnishing": "furnishing",
    "Property Size": "property_size",
    "24-hour Electricity": "24_hour_electricity",
    "Air Conditioning": "air_conditioning",
    "bedrooms": "bedrooms",
    "bathrooms": "bathrooms",
    "price": "price",
    "house_type": "house_type",
    "url": "url",
    "fetch_date": "fetch_date",
    "Apartment": "apartment",
    "Balcony": "balcony",
    "Chandelier": "chandelier",
    "Dining Area": "dining_area",
    "Dishwasher": "dishwasher",
    "Hot Water": "hot_water",
    "Kitchen Cabinets": "kitchen_cabinets",
    "Kitchen Shelf": "kitchen_shelf",
    "Microwave": "microwave",
    "Pop Ceiling": "pop_ceiling",
    "Pre-Paid Meter": "pre_paid_meter",
    "Refrigerator": "refrigerator",
    "TV": "tv",
    "Tiled Floor": "tiled_floor",
    "Wardrobe": "wardrobe",
    "Wi-Fi": "wi_fi",
}

AMENITY_COLUMNS = [
    "24_hour_electricity",
    "air_conditioning",
    "apartment",
    "balcony",
    "chandelier",
    "dining_area",
    "dishwasher",
    "hot_water",
    "kitchen_cabinets",
    "kitchen_shelf",
    "microwave",
    "pop_ceiling",
    "pre_paid_meter",
    "refrigerator",
    "tv",
    "tiled_floor",
    "wardrobe",
    "wi_fi",
]

LUXURY_AMENITIES = [
    "air_conditioning",
    "chandelier",
    "dishwasher",
    "hot_water",
    "microwave",
    "refrigerator",
    "tv",
    "wi_fi",
]

CATEGORICAL_COLUMNS = ["house_type", "condition", "furnishing"]
NUMERIC_COLUMNS = ["bedrooms", "bathrooms"]
DROP_COLUMNS = ["url", "fetch_date", "property_size", "loc"]
