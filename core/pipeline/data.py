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
from core.config import load_schema
from core.logger import logger

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


def _clean_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data = _rename_columns(data)
    drop_list = ["fetch_date", "property_size", "locality_grouped"]
    data = data.drop(
        columns=[c for c in drop_list if c in data.columns], errors="ignore"
    )
    data = _clean_strings(data)
    return data


def _filter_price_outliers(
    data: pd.DataFrame, l1: float | None = None, l2: float | None = None
) -> pd.DataFrame:
    data = data[data["price"] > 0].copy()
    x = np.log(data["price"])
    if l1 is None or l2 is None:
        q1, q3 = np.percentile(x, [25, 75])
        iqr = q3 - q1
        l1 = q1 - 1.5 * iqr
        l2 = q3 + 1.5 * iqr
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

    min_listings = 50
    locality_counts = data["locality"].value_counts()
    rare_localities = locality_counts[locality_counts < min_listings].index
    data["locality_grouped"] = data["locality"].where(
        ~data["locality"].isin(rare_localities), other="OTHER"
    )

    train_df, eval_df = train_test_split(
        data, test_size=0.2, random_state=2025, stratify=data["locality_grouped"]
    )

    train_df.to_csv(os.path.join(section["root_dir"], "train.csv"), index=False)
    eval_df.to_csv(os.path.join(section["root_dir"], "eval.csv"), index=False)
    logger.info(f"Split complete. Train: {train_df.shape}, Test: {eval_df.shape}")


def _add_lat_lng(
    data: pd.DataFrame, maps_client, geocode_cache: dict, cache_path
) -> pd.DataFrame:
    if "loc" not in data.columns:
        return data

    def get_lat_lng(location: str):
        if pd.isna(location):
            return None, None
        loc_lower = location.lower().strip()
        if loc_lower in geocode_cache:
            res = geocode_cache[loc_lower]
            return res["lat"], res["lng"]
        if maps_client is None:
            return None, None
        try:
            result = maps_client.geocode(loc_lower, region="gh")
            if result:
                lat = result[0]["geometry"]["location"]["lat"]
                lng = result[0]["geometry"]["location"]["lng"]
                geocode_cache[loc_lower] = {"lat": lat, "lng": lng}
                save_json(cache_path, geocode_cache)
                return lat, lng
        except Exception as e:
            logger.error(f"Error geocoding {location}: {e}")
        return None, None

    locations = data["loc"].dropna().astype(str).str.strip().str.lower().unique()
    loc_to_coords = {loc: get_lat_lng(loc) for loc in locations}
    coords = data["loc"].astype(str).str.strip().str.lower().map(loc_to_coords)
    data = data.copy()
    data["lat"] = coords.str[0]
    data["lng"] = coords.str[1]
    return data


def process(config: dict) -> None:
    section = config["data_processing"]
    create_directories([section["root_dir"]])

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
    l1, l2 = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    train = _filter_price_outliers(train, l1, l2)
    test = _filter_price_outliers(test, l1, l2)

    train = _clean_dataframe(train)
    test = _clean_dataframe(test)

    train = _add_lat_lng(train, maps_client, geocode_cache, section["geocode_cache"])
    test = _add_lat_lng(test, maps_client, geocode_cache, section["geocode_cache"])

    train.dropna(inplace=True)
    test.dropna(inplace=True)

    train.to_csv(
        os.path.join(section["root_dir"], "preprocessed_train.csv"), index=False
    )
    test.to_csv(os.path.join(section["root_dir"], "preprocessed_eval.csv"), index=False)

    logger.info(f"Preprocessing complete. Train: {train.shape}, Test: {test.shape}")
