"""
Feature engineering: data encoding and transformation is done here.

- Reads cleaned train/eval CSVs
- Applies feature engineering
- Saves feature-engineered CSVs
- ALSO saves fitted encoders for inference
"""

import pandas as pd
import numpy as np
import os
from ares.utils.common import load_json, save_json
from ares.entity.config_entity import FeatureEngineeringConfig
from ares import logger


class EngineerFeatures:
    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config

        self.train = pd.read_csv(self.config.train)
        self.test = pd.read_csv(self.config.test)

        self.schema = load_json(self.config.schema)
        self.geocode_cache = load_json(self.config.geocode_cache)
        self.mappings = self.schema["mappings"]
        self.lists = self.schema["lists"]

        # State variables calculated during fit, loaded during inference
        self.stats_map = {}
        self.global_ref = {}
        self.class_pi = {}
        self.loc_pi = {}

    # ---------- core transformation logic, reused during inference ----------

    def run_pipeline(self, df: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
        """Entry point for transforming any dataframe."""
        data = df.copy()

        data[["lat", "lng"]] = data["loc"].apply(
            lambda x: pd.Series(self._get_lat_lng(x))
        )

        data = self._apply_geo_features(data)

        # Target transformation if available
        if "price" in data.columns:
            data["log_price"] = np.log(data["price"])

        data = self._add_amenity_features(data)
        data = self._add_unit_density(data)

        data = self._map_locality_class(data)

        data["class_pi"] = data["loc_class"].map(self.class_pi)
        data["loc_pi"] = data["locality"].map(self.loc_pi)

        # Map Bayesian stats
        for stat in ["loc_std_dev", "loc_trust_score", "loc_tier_code"]:
            data[stat] = data["locality"].map(
                lambda x: self.stats_map.get(x, {}).get(stat, np.nan)
            )

        data["condition"] = data["condition"].map(self.mappings["condition_transform"])
        data["furnishing"] = data["furnishing"].map(
            self.mappings["furnishing_transform"]
        )

        return self.__finalize_columns(data)

    # ---------- fitting logic, applied to train only ----------

    def fit_and_save_stats(self):
        """Calculates statistics from training data and saves them."""
        self.train["log_price"] = np.log(self.train["price"])

        # Calculate Global References
        self.global_ref = {
            "median": self.train["log_price"].median(),
            "std": self.train["log_price"].std(),
        }

        self.class_pi = self.train.groupby("loc_class")["log_price"].median().to_dict()
        self.loc_pi = self.train.groupby("locality")["log_price"].median().to_dict()

        # Build Smoothed Locality Stats
        locality_agg = (
            self.train.groupby("locality")["log_price"]
            .agg(n_listings="size", std_log="std")
            .reset_index()
        )

        self.stats_map = {
            row["locality"]: self._compute_bayesian_stats(row)
            for _, row in locality_agg.iterrows()
        }

        save_json(self.config.root_dir / "locality_stats.json", self.stats_map)
        save_json(self.config.root_dir / "global_ref.json", self.global_ref)
        save_json(self.config.root_dir / "class_pi.json", self.class_pi)

    # ---------- functions for internal use only ----------

    def _get_lat_lng(self, location):
        if pd.isna(location):
            return (None, None)
        loc_data = self.geocode_cache.get(location.lower())
        return (loc_data["lat"], loc_data["lng"]) if loc_data else (None, None)

    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Vectorized Haversine calculation for DF columns or scalars."""
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        return 6371 * c

    def _apply_geo_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculates rotations and hub distances"""

        data["rot_45_lat"] = 0.707 * data["lat"] + 0.707 * data["lng"]
        data["rot_45_lng"] = 0.707 * data["lng"] - 0.707 * data["lat"]

        HUBS = {
            "airport": (5.605, -0.166),
            "accra_mall": (5.620, -0.173),
            "cbd": (5.550, -0.201),
            "east_legon_center": (5.632, -0.150),
        }

        for hub_name, coords in HUBS.items():
            data[f"dist_to_{hub_name}"] = self._haversine_distance(
                data["lat"], data["lng"], coords[0], coords[1]
            )

        hub_cols = [f"dist_to_{h}" for h in HUBS.keys()]
        data["min_dist_to_hub"] = data[hub_cols].min(axis=1)

        return data

    def _map_locality_class(self, data: pd.DataFrame):
        data["loc_class"] = (
            data["locality"].map(self.mappings["location_class"]).fillna("other")
        )

        data["loc_class"] = pd.Categorical(
            data["loc_class"], categories=self.lists["location_classes"]
        )

        return data

    def _add_amenity_features(self, data: pd.DataFrame):
        lux = self.lists["amenities"]["luxury"]
        std = self.lists["amenities"]["standard"]

        data["luxury_score"] = data[lux].sum(axis=1)
        data["amenity_count"] = data[lux].sum(axis=1) + data[std].sum(axis=1)

        drop_cols = [c for c in (lux + std) if c in data.columns]
        return data.drop(columns=drop_cols)

    def _add_unit_density(self, data: pd.DataFrame):
        data["unit_density"] = data["house_type"].map(self.mappings["property_density"])
        return data

    def _compute_bayesian_stats(self, row, K=50):
        """Logic for smoothing. Used during fit."""
        loc = row["locality"]
        n = row["n_listings"]

        w = 1.0 if loc in self.lists["elite_areas"] else n / (n + K)

        tier = 3 if w >= 0.75 else 2 if w >= 0.50 else 1 if w >= 0.25 else 0

        local_std = (
            row["std_log"] if pd.notnull(row["std_log"]) else self.global_ref["std"]
        )
        smoothed_std = w * local_std + (1 - w) * self.global_ref["std"]

        return {
            "loc_std_dev": smoothed_std,
            "loc_trust_score": w,
            "loc_tier_code": tier,
        }

    def __finalize_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Ensures the dataframe has exactly the columns required by the model,
        in the exact order specified in the schema.
        """
        required = self.lists["required_columns"]

        # Add missing columns as 0/NaN
        for col in required:
            if col not in data.columns:
                logger.warning(
                    f"Feature {col} missing during transform. Filling with 0."
                )
                data[col] = 0

        return data[required]

    def transform(self):
        """Feature egineerinfg for training pipeline."""
        self.train = self._map_locality_class(self.train)
        self.fit_and_save_stats()

        # Transform both
        self.train = self.run_pipeline(self.train)
        self.test = self.run_pipeline(self.test)

        # Save
        self.train.to_csv(
            os.path.join(self.config.root_dir, "features_train.csv"), index=False
        )
        self.test.to_csv(
            os.path.join(self.config.root_dir, "features_test.csv"), index=False
        )

        logger.info(
            f"Pipeline complete. Train: {self.train.shape}, Test: {self.test.shape}"
        )
