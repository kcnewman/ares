import os

import numpy as np
import pandas as pd

from ares import logger
from ares.entity.config_entity import FeatureEngineeringConfig
from ares.utils.common import load_json, save_json
from ares.utils.volatility import (
    compute_log_iqr,
    derive_volatility_thresholds,
    shrink_to_global,
)


class EngineerFeatures:
    def __init__(self, config: FeatureEngineeringConfig, mode: str = "train"):
        self.config = config
        self.mode = mode

        if mode == "train":
            self.train = pd.read_csv(self.config.train)
            self.test = pd.read_csv(self.config.test)

        self.schema = load_json(self.config.schema)
        self.geocode_cache = load_json(self.config.geocode_cache)
        self.mappings = self.schema["mappings"]
        self.lists = self.schema["lists"]

        # Constants
        self.DEFAULT_COORDS = (5.550, -0.201)
        self.AIRPORT_COORDS = (5.605, -0.166)
        self.K_SMOOTHING = 50
        self.ELITE_LUX_THRESHOLD = 5
        self.ELITE_LOC_PI_THRESHOLD = 0.8

        self.lat_map = {
            k: v.get("lat", self.DEFAULT_COORDS[0])
            for k, v in self.geocode_cache.items()
        }
        self.lng_map = {
            k: v.get("lng", self.DEFAULT_COORDS[1])
            for k, v in self.geocode_cache.items()
        }

        # State variables
        if mode == "inference":
            self._load_stats()
        else:
            self.stats_map = {}
            self.global_ref = {}
            self.class_pi = {}
            self.loc_pi = {}
            self.loc_luxury_median = {}
            self.global_lux_median = 0

    def run_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Entry point for transforming any dataframe (Train or Inference)."""
        data = df.copy()

        clean_loc = data["loc"].str.lower()
        data["lat"] = clean_loc.map(self.lat_map).fillna(self.DEFAULT_COORDS[0])
        data["lng"] = clean_loc.map(self.lng_map).fillna(self.DEFAULT_COORDS[1])

        data = self._apply_geo_features(data)
        data = self._add_amenity_features(data)
        data = self._add_unit_density(data)
        data = self._map_locality_class(data)

        data["class_pi"] = (
            data["loc_class"]
            .map(self.class_pi)
            .fillna(self.global_ref.get("median", 0))
        )
        data["loc_pi"] = (
            data["loc"].map(self.loc_pi).fillna(self.global_ref.get("median", 0))
        )

        data["loc_price_volatility"] = (
            data["loc"].map(self.loc_iqr).fillna(self.global_ref.get("iqr", 0.5))
        )

        data = self._add_elite_features(data)

        data["loc_std_dev"] = (
            data["loc"].map(self.stats_map).fillna(self.global_ref.get("std", 0))
        )

        data["condition"] = data["condition"].map(self.mappings["condition_transform"])
        data["furnishing"] = data["furnishing"].map(
            self.mappings["furnishing_transform"]
        )

        if "price" in data.columns:
            data["log_price"] = np.log(data["price"])

        return self.__finalize_columns(data)

    def fit_and_save_stats(self):
        """Calculates statistics from training data."""
        self.train["log_price"] = np.log(self.train["price"])
        global_iqr = compute_log_iqr(self.train["log_price"])

        self.global_ref = {
            "median": self.train["log_price"].median(),
            "std": self.train["log_price"].std(),
            "iqr": global_iqr,
        }

        self.class_pi = self.train.groupby("loc_class")["log_price"].median().to_dict()
        self.loc_pi = self.train.groupby("loc")["log_price"].median().to_dict()

        self.global_lux_median = self.train["luxury_score"].median()
        self.loc_luxury_median = (
            self.train.groupby("loc")["luxury_score"].median().to_dict()
        )

        locality_agg = (
            self.train.groupby("loc")["log_price"]
            .agg(
                n_listings="size",
                std_log="std",
                iqr_log=compute_log_iqr,
            )
            .reset_index()
        )

        global_std = self.global_ref["std"]
        locality_agg["loc_std_dev"] = locality_agg.apply(
            lambda row: shrink_to_global(
                local_value=float(row["std_log"])
                if pd.notna(row["std_log"])
                else float(global_std),
                n_listings=float(row["n_listings"]),
                global_value=float(global_std),
                k_smoothing=float(self.K_SMOOTHING),
                force_full_weight=bool(
                    "elite_areas" in self.lists
                    and row["loc"] in self.lists["elite_areas"]
                ),
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
                k_smoothing=float(self.K_SMOOTHING),
                force_full_weight=bool(
                    "elite_areas" in self.lists
                    and row["loc"] in self.lists["elite_areas"]
                ),
            ),
            axis=1,
        )

        self.stats_map = locality_agg.set_index("loc")["loc_std_dev"].to_dict()
        self.loc_iqr = locality_agg.set_index("loc")["loc_iqr"].to_dict()

        train_loc_vol = self.train["loc"].map(self.loc_iqr).fillna(global_iqr)
        vol_q25, vol_q75 = derive_volatility_thresholds(train_loc_vol)
        self.global_ref["volatility_q25"] = vol_q25
        self.global_ref["volatility_q75"] = vol_q75

        save_json(self.config.root_dir / "locality_stats.json", self.stats_map)
        save_json(self.config.root_dir / "class_pi.json", self.class_pi)
        save_json(self.config.root_dir / "loc_pi.json", self.loc_pi)
        save_json(self.config.root_dir / "loc_iqr.json", self.loc_iqr)
        save_json(
            self.config.root_dir / "loc_luxury_median.json", self.loc_luxury_median
        )
        save_json(self.config.root_dir / "global_ref.json", self.global_ref)
        save_json(
            self.config.root_dir / "global_lux_median.json",
            {"global_lux_median": self.global_lux_median},
        )

    # ---------- Internal Helper Functions ----------

    def _load_stats(self):
        """Load pre-fitted statistics for inference."""
        self.stats_map = load_json(self.config.root_dir / "locality_stats.json")
        self.class_pi = load_json(self.config.root_dir / "class_pi.json")
        self.loc_luxury_median = load_json(
            self.config.root_dir / "loc_luxury_median.json"
        )
        self.loc_pi = load_json(self.config.root_dir / "loc_pi.json")
        self.loc_iqr = load_json(self.config.root_dir / "loc_iqr.json")
        self.global_ref = load_json(self.config.root_dir / "global_ref.json")
        self.global_lux_median = load_json(
            self.config.root_dir / "global_lux_median.json"
        )
        self.global_lux_median = self.global_lux_median["global_lux_median"]

    def _apply_geo_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculates distance-to-hubs."""
        data["dist_to_airport"] = self._haversine_distance(
            data["lat"], data["lng"], self.AIRPORT_COORDS[0], self.AIRPORT_COORDS[1]
        )
        return data

    def _add_elite_features(self, data: pd.DataFrame) -> pd.DataFrame:
        data["rel_luxury"] = data["luxury_score"] - data["loc"].map(
            self.loc_luxury_median
        ).fillna(self.global_lux_median)

        data["size_density_idx"] = data["bedrooms"] * data["unit_density"]

        data["class_luxury_premium"] = data["rel_luxury"] * data["class_pi"]

        data["is_elite_tier"] = (
            (data["luxury_score"] > self.ELITE_LUX_THRESHOLD)
            & (data["loc_pi"] > self.ELITE_LOC_PI_THRESHOLD)
        ).astype(int)

        return data

    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        d = 2 * np.arcsin(
            np.sqrt(
                np.sin((lat2 - lat1) / 2) ** 2
                + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
            )
        )
        return 6371 * d

    def _map_locality_class(self, data: pd.DataFrame):
        if "loc_class" in data.columns:
            return data
        data["loc_class"] = (
            data["loc"].map(self.mappings["location_class"]).fillna("other")
        )
        return data

    def _add_amenity_features(self, data: pd.DataFrame):
        if "luxury_score" in data.columns:
            return data
        lux = self.lists["amenities"]["luxury"]
        data["luxury_score"] = data[[c for c in lux if c in data.columns]].sum(axis=1)
        return data

    def _add_unit_density(self, data: pd.DataFrame):
        data["unit_density"] = data["house_type"].map(self.mappings["property_density"])
        return data

    def __finalize_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Keep only required columns"""
        required = self.lists["required_columns"].copy()

        if self.mode == "inference" and "log_price" in required:
            required.remove("log_price")

        return data[[col for col in required if col in data.columns]]

    def transform(self):
        """Trigger for training pipeline."""
        self.train["log_price"] = np.log(self.train["price"])
        self.train = self._map_locality_class(self.train)
        self.train = self._add_amenity_features(self.train)

        self.fit_and_save_stats()

        self.train = self.run_pipeline(self.train)
        self.test = self.run_pipeline(self.test)

        self.train.to_csv(
            os.path.join(self.config.root_dir, "features_train.csv"), index=False
        )
        self.test.to_csv(
            os.path.join(self.config.root_dir, "features_test.csv"), index=False
        )
        logger.info(
            f"Pipeline complete. Train: {self.train.shape}, Test: {self.test.shape}"
        )
