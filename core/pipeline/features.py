from pathlib import Path

import numpy as np
import pandas as pd

from core.common import create_directories, load_json, save_json
from core.logger import logger
from core.volatility import (
    compute_log_iqr,
    derive_volatility_thresholds,
    shrink_to_global,
)

DEFAULT_COORDS = (5.550, -0.201)
AIRPORT_COORDS = (5.605, -0.166)
K_SMOOTHING = 50
ELITE_LUX_THRESHOLD = 5
ELITE_LOC_PI_THRESHOLD = 0.8


def _build_pipeline_dict(
    schema: dict,
    geocode_cache: dict,
    lat_map: dict,
    lng_map: dict,
    global_ref: dict,
    class_pi: dict,
    loc_pi: dict,
    loc_iqr: dict,
    stats_map: dict,
    loc_luxury_median: dict,
    global_lux_median: float | dict,
) -> dict:
    return {
        "schema": schema,
        "geocode_cache": geocode_cache,
        "lat_map": lat_map,
        "lng_map": lng_map,
        "global_ref": global_ref,
        "class_pi": class_pi,
        "loc_pi": loc_pi,
        "loc_iqr": loc_iqr,
        "stats_map": stats_map,
        "loc_luxury_median": loc_luxury_median,
        "global_lux_median": global_lux_median,
    }


def fit_features(config: dict) -> None:
    section = config["feature_engineering"]
    root_dir = Path(section["root_dir"])
    create_directories([root_dir])

    train = pd.read_csv(section["train"])
    test = pd.read_csv(section["test"])
    schema = load_json(section["schema"])
    geocode_cache = load_json(section["geocode_cache"])

    lat_map = {k: v.get("lat", DEFAULT_COORDS[0]) for k, v in geocode_cache.items()}
    lng_map = {k: v.get("lng", DEFAULT_COORDS[1]) for k, v in geocode_cache.items()}

    train["log_price"] = np.log(train["price"])
    train = _map_locality_class(train, schema)
    train = _add_amenity_features(train, schema)

    global_iqr = compute_log_iqr(train["log_price"])
    global_ref = {
        "median": train["log_price"].median(),
        "std": train["log_price"].std(),
        "iqr": global_iqr,
    }

    class_pi = train.groupby("loc_class")["log_price"].median().to_dict()
    loc_pi = train.groupby("loc")["log_price"].median().to_dict()
    global_lux_median = train["luxury_score"].median()
    loc_luxury_median = train.groupby("loc")["luxury_score"].median().to_dict()

    locality_agg = (
        train.groupby("loc")["log_price"]
        .agg(n_listings="size", std_log="std", iqr_log=compute_log_iqr)
        .reset_index()
    )

    elite_areas = schema.get("lists", {}).get("elite_areas", [])
    global_std = global_ref["std"]

    locality_agg["loc_std_dev"] = locality_agg.apply(
        lambda row: shrink_to_global(
            local_value=float(row["std_log"])
            if pd.notna(row["std_log"])
            else float(global_std),
            n_listings=float(row["n_listings"]),
            global_value=float(global_std),
            k_smoothing=K_SMOOTHING,
            force_full_weight=row["loc"] in elite_areas,
        ),
        axis=1,
    )
    locality_agg["loc_iqr"] = locality_agg.apply(
        lambda row: shrink_to_global(
            local_value=float(row["iqr_log"])
            if pd.notna(row["iqr_log"])
            else float(global_iqr),
            n_listings=float(row["n_listings"]),
            global_value=float(global_iqr),
            k_smoothing=K_SMOOTHING,
            force_full_weight=row["loc"] in elite_areas,
        ),
        axis=1,
    )

    stats_map = locality_agg.set_index("loc")["loc_std_dev"].to_dict()
    loc_iqr = locality_agg.set_index("loc")["loc_iqr"].to_dict()

    train_loc_vol = train["loc"].map(loc_iqr).fillna(global_iqr)
    vol_q25, vol_q75 = derive_volatility_thresholds(train_loc_vol)
    global_ref["volatility_q25"] = vol_q25
    global_ref["volatility_q75"] = vol_q75

    save_json(root_dir / "locality_stats.json", stats_map)
    save_json(root_dir / "class_pi.json", class_pi)
    save_json(root_dir / "loc_pi.json", loc_pi)
    save_json(root_dir / "loc_iqr.json", loc_iqr)
    save_json(root_dir / "loc_luxury_median.json", loc_luxury_median)
    save_json(root_dir / "global_ref.json", global_ref)
    save_json(
        root_dir / "global_lux_median.json", {"global_lux_median": global_lux_median}
    )

    pipeline = _build_pipeline_dict(
        schema,
        geocode_cache,
        lat_map,
        lng_map,
        global_ref,
        class_pi,
        loc_pi,
        loc_iqr,
        stats_map,
        loc_luxury_median,
        global_lux_median,
    )
    train_features = transform_features(train, pipeline)
    train_features["log_price"] = train["log_price"]

    test_features = transform_features(test, pipeline)

    train_features.to_csv(root_dir / "features_train.csv", index=False)
    test_features.to_csv(root_dir / "features_test.csv", index=False)

    logger.info(
        f"Feature engineering complete. Train: {train_features.shape}, Test: {test_features.shape}"
    )


def make_feature_pipeline(config: dict) -> dict:
    section = config["feature_engineering"]
    root_dir = section["root_dir"]
    schema = load_json(section["schema"])
    geocode_cache = load_json(section["geocode_cache"])

    return _build_pipeline_dict(
        schema,
        geocode_cache,
        {k: v.get("lat", DEFAULT_COORDS[0]) for k, v in geocode_cache.items()},
        {k: v.get("lng", DEFAULT_COORDS[1]) for k, v in geocode_cache.items()},
        load_json(root_dir / "global_ref.json"),
        load_json(root_dir / "class_pi.json"),
        load_json(root_dir / "loc_pi.json"),
        load_json(root_dir / "loc_iqr.json"),
        load_json(root_dir / "locality_stats.json"),
        load_json(root_dir / "loc_luxury_median.json"),
        load_json(root_dir / "global_lux_median.json")["global_lux_median"],
    )


def transform_features(df: pd.DataFrame, pipeline: dict) -> pd.DataFrame:
    data = df.copy()
    schema = pipeline["schema"]

    clean_loc = data["loc"].str.lower()
    data["lat"] = clean_loc.map(pipeline["lat_map"]).fillna(DEFAULT_COORDS[0])
    data["lng"] = clean_loc.map(pipeline["lng_map"]).fillna(DEFAULT_COORDS[1])

    data = _apply_geo_features(data)
    data = _add_amenity_features(data, schema)
    data = _add_unit_density(data, schema)
    data = _map_locality_class(data, schema)

    data["class_pi"] = (
        data["loc_class"]
        .map(pipeline["class_pi"])
        .fillna(pipeline["global_ref"].get("median", 0))
    )
    data["loc_pi"] = (
        data["loc"]
        .map(pipeline["loc_pi"])
        .fillna(pipeline["global_ref"].get("median", 0))
    )
    data["loc_price_volatility"] = (
        data["loc"]
        .map(pipeline["loc_iqr"])
        .fillna(pipeline["global_ref"].get("iqr", 0.5))
    )
    data["loc_std_dev"] = (
        data["loc"]
        .map(pipeline["stats_map"])
        .fillna(pipeline["global_ref"].get("std", 0))
    )

    data = _add_elite_features(data, pipeline)
    data["condition"] = data["condition"].map(schema["mappings"]["condition_transform"])
    data["furnishing"] = data["furnishing"].map(
        schema["mappings"]["furnishing_transform"]
    )

    if "price" in data.columns:
        data["log_price"] = np.log(data["price"])

    return _finalize_columns(data, schema)


def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    d = 2 * np.arcsin(
        np.sqrt(
            np.sin((lat2 - lat1) / 2) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
        )
    )
    return 6371 * d


def _apply_geo_features(data: pd.DataFrame) -> pd.DataFrame:
    data["dist_to_airport"] = _haversine_distance(
        data["lat"], data["lng"], AIRPORT_COORDS[0], AIRPORT_COORDS[1]
    )
    return data


def _add_amenity_features(data: pd.DataFrame, schema: dict) -> pd.DataFrame:
    if "luxury_score" in data.columns:
        return data
    lux = schema["lists"]["amenities"]["luxury"]
    data["luxury_score"] = data[[c for c in lux if c in data.columns]].sum(axis=1)
    return data


def _add_unit_density(data: pd.DataFrame, schema: dict) -> pd.DataFrame:
    if "house_type" not in data.columns:
        return data
    data["unit_density"] = data["house_type"].map(
        schema["mappings"]["property_density"]
    )
    return data


def _map_locality_class(data: pd.DataFrame, schema: dict) -> pd.DataFrame:
    if "loc_class" in data.columns:
        return data
    data["loc_class"] = (
        data["loc"].map(schema["mappings"]["location_class"]).fillna("other")
    )
    return data


def _add_elite_features(data: pd.DataFrame, pipeline: dict) -> pd.DataFrame:
    data["rel_luxury"] = data["luxury_score"] - data["loc"].map(
        pipeline["loc_luxury_median"]
    ).fillna(pipeline["global_lux_median"])

    data["size_density_idx"] = data["bedrooms"] * data["unit_density"]
    data["class_luxury_premium"] = data["rel_luxury"] * data["class_pi"]

    data["is_elite_tier"] = (
        (data["luxury_score"] > ELITE_LUX_THRESHOLD)
        & (data["loc_pi"] > ELITE_LOC_PI_THRESHOLD)
    ).astype(int)

    return data


def _finalize_columns(data: pd.DataFrame, schema: dict) -> pd.DataFrame:
    required = schema["lists"]["required_columns"].copy()
    if "log_price" in required:
        required.remove("log_price")
    return data[[col for col in required if col in data.columns]]
