import json
import os

import googlemaps
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pandas.api.types import (
    is_bool_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_object_dtype,
    is_string_dtype,
)
from sklearn.model_selection import train_test_split

from core.common import create_directories, load_json, save_json
from core.config import load_config, load_schema
from core.logger import logger

DEFAULT_COORDS = (5.550, -0.201)

load_dotenv()


def _build_maps_client() -> googlemaps.Client | None:
    api_key = os.getenv("GOOGLE_MAPS_KEY")
    if not api_key:
        logger.warning(
            "GOOGLE_MAPS_KEY is not configured. Geocoding fallback will be skipped."
        )
        return None
    try:
        return googlemaps.Client(key=api_key)
    except Exception as exc:
        logger.error("Failed to initialize Google Maps client: %s", exc)
        return None


def _is_dtype_compatible(series: pd.Series, expected_dtype: str) -> bool:
    expected = expected_dtype.strip().lower()
    if expected.startswith("float"):
        return bool(is_float_dtype(series))
    if expected.startswith("int"):
        return bool(is_integer_dtype(series))
    if expected in {"object", "string", "str"}:
        return bool(is_object_dtype(series) or is_string_dtype(series))
    if expected.startswith("bool"):
        return bool(is_bool_dtype(series))
    return str(series.dtype) == expected


def _clean_strings(data: pd.DataFrame) -> pd.DataFrame:
    string_cols = data.select_dtypes(include=["object"]).columns
    for col in string_cols:
        if col == "url":
            continue
        data[col] = (
            data[col]
            .astype(str)
            .str.replace(r"[^a-zA-Z]", " ", regex=True)
            .str.lower()
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )
    data.replace(["nan", ""], np.nan, inplace=True)
    return data


def _rename_columns(data: pd.DataFrame) -> pd.DataFrame:
    new_cols = {
        col: col.lower().replace(" ", "_").replace("-", "_") for col in data.columns
    }
    return data.rename(columns=new_cols)


def _clean_dataframe(data: pd.DataFrame, drop_list: list[str] | None = None) -> pd.DataFrame:
    data = data.copy()
    data = _rename_columns(data)
    drop_list = drop_list or ["fetch_date", "property_size", "locality_grouped"]
    data = data.drop(
        columns=[c for c in drop_list if c in data.columns], errors="ignore"
    )
    data = _clean_strings(data)
    return data


def _filter_price_outliers(
    data: pd.DataFrame, l1: float | None = None, l2: float | None = None,
    multiplier: float = 1.5,
) -> pd.DataFrame:
    data = data[data["price"] > 0].copy()
    x = np.log(data["price"])
    if l1 is None or l2 is None:
        q1, q3 = np.percentile(x, [25, 75])
        iqr = q3 - q1
        l1 = q1 - multiplier * iqr
        l2 = q3 + multiplier * iqr
    return data[(x >= l1) & (x <= l2)]


def validate(config: dict) -> bool:
    section = config["data_validation"]
    schema = load_schema()["COLUMNS"]
    create_directories([section["root_dir"]])

    try:
        status = True
        error_msg = "All checks passed"

        data = pd.read_csv(section["data_dir"])
        all_cols = list(data.columns)

        for col in schema:
            if col not in all_cols:
                status = False
                error_msg = f"Missing column: {col}"
                break

        if status:
            for col in all_cols:
                if col not in schema:
                    status = False
                    error_msg = f"Unexpected column: {col}"
                    break

        if status:
            for col in all_cols:
                actual_dtype = str(data[col].dtype)
                expected_dtype = schema[col]
                if not _is_dtype_compatible(data[col], expected_dtype):
                    status = False
                    error_msg = f"Type mismatch for '{col}': Expected {expected_dtype}, got {actual_dtype}"
                    break

        with open(section["status_file"], "w") as f:
            json.dump({"passed": status, "detail": error_msg}, f)

        return status

    except Exception as e:
        with open(section["status_file"], "w") as f:
            json.dump({"passed": False, "detail": f"Exception occurred: {str(e)}"}, f)
        raise


def split(config: dict) -> None:
    section = config["data_split"]
    status_file = section["status_file"]
    create_directories([section["root_dir"]])

    if not os.path.exists(status_file):
        raise FileNotFoundError(f"Validation status file missing at {status_file}.")

    with open(status_file) as f:
        status = json.load(f)
        if not status.get("passed", False):
            raise RuntimeError(
                f"Data Validation failed: {status.get('detail', 'Unknown error')}"
            )

    logger.info("Validation passed. Starting data split...")
    data = pd.read_csv(section["data_dir"])

    min_listings = config.get("data_split", {}).get("min_listings", 50)
    locality_counts = data["locality"].value_counts()
    rare_localities = locality_counts[locality_counts < min_listings].index
    data["locality_grouped"] = data["locality"].where(
        ~data["locality"].isin(rare_localities), other="OTHER"
    )

    test_size = config.get("data_split", {}).get("test_size", 0.2)
    random_state = config.get("data_split", {}).get("random_state", 2025)
    train_df, eval_df = train_test_split(
        data, test_size=test_size, random_state=random_state, stratify=data["locality_grouped"]
    )

    train_df.to_csv(os.path.join(section["root_dir"], "train.csv"), index=False)
    eval_df.to_csv(os.path.join(section["root_dir"], "eval.csv"), index=False)
    logger.info(f"Split complete. Train: {train_df.shape}, Test: {eval_df.shape}")


def _add_lat_lng(
    data: pd.DataFrame, maps_client, geocode_cache: dict, cache_path
) -> pd.DataFrame:
    if "loc" not in data.columns:
        return data

    locations = data["loc"].dropna().astype(str).str.strip().str.lower().unique()
    loc_to_coords = {}
    dirty = False

    for loc in locations:
        if loc in geocode_cache:
            res = geocode_cache[loc]
            loc_to_coords[loc] = (res["lat"], res["lng"])
            continue
        if maps_client is None:
            loc_to_coords[loc] = (None, None)
            continue
        try:
            result = maps_client.geocode(loc, region="gh")
            if result:
                lat = result[0]["geometry"]["location"]["lat"]
                lng = result[0]["geometry"]["location"]["lng"]
                geocode_cache[loc] = {"lat": lat, "lng": lng}
                loc_to_coords[loc] = (lat, lng)
                dirty = True
                continue
        except Exception as e:
            logger.error(f"Error geocoding {loc}: {e}")
        loc_to_coords[loc] = (None, None)

    if dirty:
        save_json(cache_path, geocode_cache)

    coords = data["loc"].astype(str).str.strip().str.lower().map(loc_to_coords)
    data = data.copy()
    data["lat"] = coords.str[0]
    data["lng"] = coords.str[1]
    return data


AMENITY_KEYWORDS = [
    "electricity", "air_conditioning", "apartment", "balcony", "chandelier",
    "dining_area", "dishwasher", "hot_water", "kitchen_cabinets", "kitchen_shelf",
    "microwave", "pop_ceiling", "pre_paid_meter", "refrigerator", "tv",
    "tiled_floor", "wardrobe", "wi_fi",
]

LUXURY_AMENITIES = {
    "air_conditioning", "chandelier", "dishwasher", "hot_water", "microwave",
    "refrigerator", "tv", "wi_fi",
}


def _build_runtime_schema(train: pd.DataFrame) -> dict:
    schema = load_schema()
    columns = {k.lower().replace(" ", "_").replace("-", "_"): v for k, v in schema["COLUMNS"].items()}

    base = {"COLUMNS": columns, "TARGET_COLUMN": schema["TARGET_COLUMN"]}

    locations = sorted(train["loc"].dropna().unique()) if "loc" in train.columns else []
    house_types = sorted(train["house_type"].dropna().unique()) if "house_type" in train.columns else []
    conditions = sorted(train["condition"].dropna().unique()) if "condition" in train.columns else []
    furnish_opts = sorted(train["furnishing"].dropna().unique()) if "furnishing" in train.columns else []

    cond_map = {v: i for i, v in enumerate(conditions)}
    furn_map = {v: i for i, v in enumerate(furnish_opts)}
    density_map = {t: 1 for t in house_types}

    amenity_cols = [c for c in columns if any(kw in c for kw in AMENITY_KEYWORDS)]
    lux_cols = [c for c in amenity_cols if c in LUXURY_AMENITIES]
    std_cols = [c for c in amenity_cols if c not in LUXURY_AMENITIES]

    feature_cols = [c for c in train.columns if c not in ("price", "url", "fetch_date", "locality", "Property Size", "property_size", "locality_grouped")]

    location_class = {loc: "other" for loc in locations}

    base["mappings"] = {
        "location_class": location_class,
        "condition_transform": cond_map,
        "furnishing_transform": furn_map,
        "property_density": density_map,
    }
    base["lists"] = {
        "amenities": {"luxury": lux_cols, "standard": std_cols},
        "required_columns": feature_cols,
        "elite_areas": [],
    }
    return base


def process(config: dict) -> None:
    section = config["data_processing"]
    create_directories([section["root_dir"]])
    create_directories(["artifacts/cache"])

    train = pd.read_csv(section["train"])
    test = pd.read_csv(section["test"])
    maps_client = _build_maps_client()
    try:
        geocode_cache = load_json(section["geocode_cache"])
    except Exception:
        geocode_cache = {}

    train.drop_duplicates(subset="url", inplace=True)
    test.drop_duplicates(subset="url", inplace=True)

    train_pos = train[train["price"] > 0]
    x = np.log(train_pos["price"])
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    multiplier = config.get("data_processing", {}).get("outlier_iqr_multiplier", 1.5)
    l1, l2 = q1 - multiplier * iqr, q3 + multiplier * iqr
    train = _filter_price_outliers(train, l1, l2, multiplier=multiplier)
    test = _filter_price_outliers(test, l1, l2, multiplier=multiplier)

    drop_columns = config.get("data_processing", {}).get("drop_columns", ["fetch_date", "property_size", "locality_grouped"])
    train = _clean_dataframe(train, drop_list=drop_columns)
    test = _clean_dataframe(test, drop_list=drop_columns)

    train = _add_lat_lng(train, maps_client, geocode_cache, section["geocode_cache"])
    test = _add_lat_lng(test, maps_client, geocode_cache, section["geocode_cache"])

    if "lat" in train.columns:
        train["lat"] = train["lat"].fillna(DEFAULT_COORDS[0])
        train["lng"] = train["lng"].fillna(DEFAULT_COORDS[1])
    if "lat" in test.columns:
        test["lat"] = test["lat"].fillna(DEFAULT_COORDS[0])
        test["lng"] = test["lng"].fillna(DEFAULT_COORDS[1])

    train.dropna(inplace=True)
    test.dropna(inplace=True)

    runtime_schema = _build_runtime_schema(train)
    save_json("artifacts/cache/schema.json", runtime_schema)

    train.to_csv(
        os.path.join(section["root_dir"], "preprocessed_train.csv"), index=False
    )
    test.to_csv(os.path.join(section["root_dir"], "preprocessed_eval.csv"), index=False)

    logger.info(f"Preprocessing complete. Train: {train.shape}, Test: {test.shape}")
