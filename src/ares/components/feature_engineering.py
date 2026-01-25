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

        # State variables for statistics
        self.stats_map = {}
        self.global_ref = {}
        self.class_pi = {}
        self.loc_pi = {}
        self.loc_luxury_median = {}
        self.loc_bed_median = {}

    def run_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Entry point for transforming any dataframe (Train or Inference)."""
        data = df.copy()

        data[["lat", "lng"]] = data["loc"].apply(
            lambda x: pd.Series(self._get_lat_lng(x))
        )

        # Transformations
        data = self._apply_geo_features(data)
        data = self._add_amenity_features(data)
        data = self._add_unit_density(data)
        data = self._map_locality_class(data)

        # Apply Fitted Pi-Encodings
        data["class_pi"] = (
            data["loc_class"]
            .map(self.class_pi)
            .fillna(self.global_ref.get("median", 0))
        )
        data["loc_pi"] = (
            data["loc"].map(self.loc_pi).fillna(self.global_ref.get("median", 0))
        )

        data = self._add_elite_features(data)

        for stat in ["loc_std_dev", "loc_trust_score", "loc_tier_code"]:
            data[stat] = data["loc"].map(
                lambda x: self.stats_map.get(x, {}).get(stat, np.nan)
            )

        data["condition"] = data["condition"].map(self.mappings["condition_transform"])
        data["furnishing"] = data["furnishing"].map(
            self.mappings["furnishing_transform"]
        )

        if "price" in data.columns:
            data["log_price"] = np.log(data["price"])

        return self.__finalize_columns(data)

    def fit_and_save_stats(self):
        """Calculates statistics from training data and saves for inference."""
        self.train["log_price"] = np.log(self.train["price"])

        self.global_ref = {
            "median": self.train["log_price"].median(),
            "std": self.train["log_price"].std(),
        }

        self.class_pi = self.train.groupby("loc_class")["log_price"].median().to_dict()
        self.loc_pi = self.train.groupby("loc")["log_price"].median().to_dict()

        self.loc_luxury_median = (
            self.train.groupby("loc")["luxury_score"].median().to_dict()
        )
        self.loc_bed_median = self.train.groupby("loc")["bedrooms"].median().to_dict()

        locality_agg = (
            self.train.groupby("loc")["log_price"]
            .agg(n_listings="size", std_log="std")
            .reset_index()
        )

        self.stats_map = {
            row["loc"]: self._compute_bayesian_stats(row)
            for _, row in locality_agg.iterrows()
        }

        # Persist stats
        save_json(self.config.root_dir / "locality_stats.json", self.stats_map)
        save_json(self.config.root_dir / "class_pi.json", self.class_pi)
        save_json(
            self.config.root_dir / "loc_luxury_median.json", self.loc_luxury_median
        )
        save_json(self.config.root_dir / "loc_bed_median.json", self.loc_bed_median)

    # ---------- Internal Helper Functions ----------

    def _apply_geo_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculates distance-to-hubs."""
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

    def _add_elite_features(self, data: pd.DataFrame) -> pd.DataFrame:
        data["bath_per_bed"] = (
            (data["bathrooms"] / data["bedrooms"])
            .replace([np.inf, -np.inf], 0)
            .fillna(0)
        )

        data["rel_luxury"] = data["luxury_score"] - data["loc"].map(
            self.loc_luxury_median
        ).fillna(0)
        data["rel_size"] = data["bedrooms"] - data["loc"].map(
            self.loc_bed_median
        ).fillna(0)

        cantonments_coord = (5.578, -0.174)
        data["dist_to_wealth_hub"] = self._haversine_distance(
            data["lat"], data["lng"], cantonments_coord[0], cantonments_coord[1]
        )

        data["is_elite_tier"] = (
            (data["luxury_score"] > 5) & (data["loc_pi"] > 0.8)
        ).astype(int)
        return data

    def _get_lat_lng(self, location):
        if pd.isna(location):
            return (None, None)
        loc_data = self.geocode_cache.get(location.lower())
        return (loc_data["lat"], loc_data["lng"]) if loc_data else (None, None)

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

    def _compute_bayesian_stats(self, row, K=50):
        n = row["n_listings"]
        w = 1.0 if row["loc"] in self.lists.get("elite_areas", []) else n / (n + K)
        local_std = (
            row["std_log"] if pd.notnull(row["std_log"]) else self.global_ref["std"]
        )
        return {
            "loc_std_dev": w * local_std + (1 - w) * self.global_ref["std"],
            "loc_trust_score": w,
            "loc_tier_code": 3 if w >= 0.75 else 2 if w >= 0.50 else 1,
        }

    def __finalize_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Keep only required columns"""
        required = self.lists["required_columns"]

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
