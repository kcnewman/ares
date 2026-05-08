from pathlib import Path

import httpx
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

LUXURY_AMENITIES = {
    "air_conditioning", "chandelier", "dishwasher", "hot_water", "microwave",
    "refrigerator", "tv", "wi_fi",
}


def _fetch_location_tiers(locations: list[str], llm_url: str) -> dict:
    try:
        resp = httpx.post(
            f"{llm_url}/classify-locations",
            json={"locations": list(set(locations))},
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json()
            return {c["location"]: {"tier": c["tier"], "is_elite": c["is_elite"]}
                    for c in data["classifications"]}
    except Exception as exc:
        logger.warning(f"LLM location classification failed: {exc}")
    return {loc.lower(): {"tier": "other", "is_elite": False} for loc in locations}


def _fetch_amenity_tiers(amenities: list[str], llm_url: str) -> dict:
    try:
        resp = httpx.post(
            f"{llm_url}/classify-amenities",
            json={"amenities": list(set(amenities))},
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json()
            return {c["amenity"]: c["tier"] for c in data["classifications"]}
    except Exception as exc:
        logger.warning(f"LLM amenity classification failed: {exc}")
    return {a: "standard" for a in amenities}


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
    elite_lux_threshold: int = 5,
    elite_loc_pi_threshold: float = 0.8,
    location_tiers: dict | None = None,
    amenity_tiers: dict | None = None,
    elite_areas: list | None = None,
    llm_service_url: str | None = None,
    root_dir: str | None = None,
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
        "elite_lux_threshold": elite_lux_threshold,
        "elite_loc_pi_threshold": elite_loc_pi_threshold,
        "location_tiers": location_tiers or {},
        "amenity_tiers": amenity_tiers or {},
        "elite_areas": elite_areas or [],
        "llm_service_url": llm_service_url,
        "root_dir": root_dir,
    }


def fit_features(config: dict) -> None:
    section = config["feature_engineering"]
    root_dir = Path(section["root_dir"])
    create_directories([root_dir])
    llm_url = config.get("llm_service_url", "http://localhost:8001")

    train = pd.read_csv(section["train"])
    test = pd.read_csv(section["test"])
    schema = load_json(section["schema"])
    try:
        geocode_cache = load_json(section["geocode_cache"])
    except Exception:
        geocode_cache = {}

    lat_map = {k: v.get("lat", DEFAULT_COORDS[0]) for k, v in geocode_cache.items()}
    lng_map = {k: v.get("lng", DEFAULT_COORDS[1]) for k, v in geocode_cache.items()}

    all_locs = pd.concat([train["loc"], test["loc"]]).unique().tolist()
    location_tiers = _fetch_location_tiers(all_locs, llm_url)

    all_other = all(v.get("tier") == "other" for v in location_tiers.values())
    if all_other and len(location_tiers) > 0:
        loc_median = train.groupby("loc")["price"].median()
        loc_median = loc_median[loc_median > 0]
        if len(loc_median) > 0:
            cuts = loc_median.quantile([0.1, 0.3, 0.5, 0.7, 0.9])
            tiers = {}
            for loc, med in loc_median.items():
                if med >= cuts[0.9]:
                    tier_name, elite = "prime", True
                elif med >= cuts[0.7]:
                    tier_name, elite = "established", False
                elif med >= cuts[0.5]:
                    tier_name, elite = "high_density", False
                elif med >= cuts[0.3]:
                    tier_name, elite = "developing_commuter", False
                elif med >= cuts[0.1]:
                    tier_name, elite = "industrial_traffic", False
                else:
                    tier_name, elite = "satellite_hub", False
                tiers[loc] = {"tier": tier_name, "is_elite": elite}
            location_tiers.update(tiers)
            logger.info("LLM unavailable — used price-quantile location tiers")

    save_json(root_dir / "location_tiers.json", location_tiers)
    logger.info(f"Classified {len(location_tiers)} locations")

    lux_amenities = schema.get("lists", {}).get("amenities", {}).get("luxury", [])
    std_amenities = schema.get("lists", {}).get("amenities", {}).get("standard", [])
    all_amenity_cols = lux_amenities + std_amenities
    amenity_tiers = _fetch_amenity_tiers(all_amenity_cols, llm_url)

    all_standard = all(v == "standard" for v in amenity_tiers.values())
    if all_standard and len(amenity_tiers) > 0:
        amenity_tiers = {a: ("luxury" if a in LUXURY_AMENITIES else "standard") for a in all_amenity_cols}
        logger.info("LLM unavailable — used hardcoded luxury amenity set")

    save_json(root_dir / "amenity_tiers.json", amenity_tiers)
    luxury_amenities = [a for a, t in amenity_tiers.items() if t == "luxury"]
    logger.info(f"Classified {len(amenity_tiers)} amenities ({len(luxury_amenities)} luxury)")

    elite_areas_override = section.get("elite_areas_override", [])
    if elite_areas_override:
        elite_areas = elite_areas_override
    else:
        elite_areas = [loc for loc, data in location_tiers.items() if data.get("is_elite")]

    train["log_price"] = np.log(train["price"])
    train = _map_locality_class(train, location_tiers)
    train = _add_amenity_features(train, luxury_amenities)

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

    k_smoothing = section.get("k_smoothing", 50)
    global_std = global_ref["std"]

    locality_agg["loc_std_dev"] = locality_agg.apply(
        lambda row: shrink_to_global(
            local_value=float(row["std_log"])
            if pd.notna(row["std_log"])
            else float(global_std),
            n_listings=float(row["n_listings"]),
            global_value=float(global_std),
            k_smoothing=k_smoothing,
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
            k_smoothing=k_smoothing,
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
        elite_lux_threshold=section.get("elite_lux_threshold", 5),
        elite_loc_pi_threshold=section.get("elite_loc_pi_threshold", 0.8),
        location_tiers=location_tiers,
        amenity_tiers=amenity_tiers,
        elite_areas=elite_areas,
        root_dir=str(root_dir),
    )
    train_features = transform_features(train, pipeline)
    test_features = transform_features(test, pipeline)

    train_features.to_csv(root_dir / "features_train.csv", index=False)
    test_features.to_csv(root_dir / "features_test.csv", index=False)

    feature_columns = [c for c in train_features.columns if c != "log_price"]
    save_json(root_dir / "feature_columns.json", {"columns": feature_columns})

    logger.info(
        f"Feature engineering complete. Train: {train_features.shape}, Test: {test_features.shape}"
    )


def make_feature_pipeline(config: dict) -> dict:
    section = config["feature_engineering"]
    root_dir = Path(section["root_dir"])
    schema = load_json(section["schema"])
    try:
        geocode_cache = load_json(section["geocode_cache"])
    except Exception:
        geocode_cache = {}
    llm_url = config.get("llm_service_url", "http://localhost:8001")

    location_tiers = {}
    amenity_tiers = {}
    try:
        location_tiers = load_json(root_dir / "location_tiers.json")
    except Exception:
        pass
    try:
        amenity_tiers = load_json(root_dir / "amenity_tiers.json")
    except Exception:
        pass

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
        elite_lux_threshold=section.get("elite_lux_threshold", 5),
        elite_loc_pi_threshold=section.get("elite_loc_pi_threshold", 0.8),
        location_tiers=location_tiers,
        amenity_tiers=amenity_tiers,
        elite_areas=section.get("elite_areas_override", []),
        llm_service_url=llm_url,
        root_dir=str(root_dir),
    )


def _resolve_unknown_locations(
    data: pd.DataFrame, location_tiers: dict, llm_url: str | None
) -> dict:
    unknown = data["loc"].str.lower().unique().tolist()
    unknown = [loc for loc in unknown if loc not in location_tiers]
    if not unknown or not llm_url:
        return {}
    fresh = _fetch_location_tiers(unknown, llm_url)
    location_tiers.update(fresh)
    return fresh


def transform_features(df: pd.DataFrame, pipeline: dict) -> pd.DataFrame:
    data = df.copy()
    schema = pipeline["schema"]
    location_tiers = pipeline.get("location_tiers", {})
    amenity_tiers = pipeline.get("amenity_tiers", {})
    luxury_amenities = [a for a, t in amenity_tiers.items() if t == "luxury"]
    llm_url = pipeline.get("llm_service_url")

    _resolve_unknown_locations(data, location_tiers, llm_url)

    clean_loc = data["loc"].str.lower()
    data["lat"] = clean_loc.map(pipeline["lat_map"]).fillna(DEFAULT_COORDS[0])
    data["lng"] = clean_loc.map(pipeline["lng_map"]).fillna(DEFAULT_COORDS[1])

    data = _apply_geo_features(data)
    data = _add_amenity_features(data, luxury_amenities)
    data = _add_unit_density(data, schema)
    data = _map_locality_class(data, location_tiers)

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


def _add_amenity_features(data: pd.DataFrame, luxury_columns: list[str]) -> pd.DataFrame:
    if "luxury_score" in data.columns:
        return data
    data["luxury_score"] = data[[c for c in luxury_columns if c in data.columns]].sum(axis=1)
    return data


def _add_unit_density(data: pd.DataFrame, schema: dict) -> pd.DataFrame:
    if "house_type" not in data.columns:
        return data
    data["unit_density"] = data["house_type"].map(
        schema["mappings"]["property_density"]
    )
    return data


TIER_ORDER = {
    "other": 0, "satellite_hub": 1, "industrial_traffic": 2,
    "developing_commuter": 3, "high_density": 4, "established": 5, "prime": 6,
}

def _map_locality_class(data: pd.DataFrame, location_tiers: dict) -> pd.DataFrame:
    if "loc_class" in data.columns:
        return data
    data["loc_class"] = (
        data["loc"]
        .str.lower()
        .map(lambda loc: TIER_ORDER.get(location_tiers.get(loc, {}).get("tier", "other"), 0))
        .fillna(0)
        .astype(int)
    )
    return data


def _add_elite_features(data: pd.DataFrame, pipeline: dict) -> pd.DataFrame:
    data["rel_luxury"] = data["luxury_score"] - data["loc"].map(
        pipeline["loc_luxury_median"]
    ).fillna(pipeline["global_lux_median"])

    data["size_density_idx"] = data["bedrooms"] * data["unit_density"]
    data["class_luxury_premium"] = data["rel_luxury"] * data["class_pi"]

    elite_lux_threshold = pipeline.get("elite_lux_threshold", 5)
    elite_loc_pi_threshold = pipeline.get("elite_loc_pi_threshold", 0.8)
    data["is_elite_tier"] = (
        (data["luxury_score"] > elite_lux_threshold)
        & (data["loc_pi"] > elite_loc_pi_threshold)
    ).astype(int)

    return data


DROP_COLS = {"price", "url", "fetch_date", "locality", "Property Size", "property_size", "locality_grouped", "loc", "house_type", "bathrooms", "bedrooms", "condition", "furnishing"}

def _finalize_columns(data: pd.DataFrame, schema: dict) -> pd.DataFrame:
    return data[[c for c in data.columns if c not in DROP_COLS]]
