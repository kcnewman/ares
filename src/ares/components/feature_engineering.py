import pandas as pd
import numpy as np
import os
from ares.utils.common import load_json, save_json
from ares.entity.config_entity import FeatureEngineeringConfig


class EngineerFeatures:
    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config
        self.train = pd.read_csv(self.config.train)
        self.test = pd.read_csv(self.config.test)
        self.locality_class = load_json(self.config.locality_class)
        self.unit_density = load_json(self.config.unit_density)
        self.stats_map = {}
        self.global_ref = {}
        self.class_pi = {}
        self.loc_pi = {}
        self.ELITE_AREAS = {
            "Airport Residential Area",
            "Cantonments",
            "Ridge",
            "Roman Ridge",
            "Labone",
        }

    def __create_log_price(self):
        """Add log-transformed price"""

        self.train["log_price"] = np.log(self.train["price"])
        self.test["log_price"] = np.log(self.test["price"])

    def __map_locality_class(self, data: pd.DataFrame):
        """Maps locality to pre-defined residential classes"""

        data["loc_class"] = data["locality"].map(self.locality_class).fillna("other")

        return data

    def __compute_global_stats(self):
        """Compute global reference statistics from training data"""

        self.global_ref = {
            "median": self.train["log_price"].median(),
            "std": self.train["log_price"].std(),
        }

    def __calculate_price_indices(self):
        """Calculate median price indices by locality class and locality"""

        class_avg = self.train.groupby("loc_class")["log_price"].median()
        self.class_pi = class_avg.to_dict()
        save_json(self.config.root_dir / "class_pi.json", self.class_pi)

        class_std = self.train.groupby("loc_class")["log_price"].std()
        class_std = class_std.to_dict()
        save_json(self.config.root_dir / "class_std.json", class_std)

        loc_avg = self.train.groupby("locality")["log_price"].median()
        self.loc_pi = loc_avg.to_dict()
        save_json(self.config.root_dir / "loc_pi.json", self.loc_pi)

    def __aggregate_by_locality(self):
        """Group by locality and compute statistics"""

        return (
            self.train.groupby("locality")["log_price"]
            .agg(
                n_listings="size",
                median_log="median",
                std_log="std",
            )
            .reset_index()
        )

    def __calculate_trust_weight(self, locality, n_listings, K=50):
        """Calculate Bayesian Smoothing weight based on size"""

        if locality in self.ELITE_AREAS:
            return 1.0
        return n_listings / (n_listings + K)

    def __get_tier_info(self, weight):
        """Map trust score to their code"""

        if weight >= 0.75:
            return 3
        if weight >= 0.50:
            return 2
        if weight >= 0.25:
            return 1
        return 0

    def __smooth_locality_stats(self, row, K=50):
        """Apply Bayesian smoothing to locality stats"""

        w = self.__calculate_trust_weight(row["locality"], row["n_listings"], K)
        tier_code = self.__get_tier_info(w)

        local_std = (
            row["std_log"] if pd.notnull(row["std_log"]) else self.global_ref["std"]
        )
        return {
            "loc_std_dev": w * local_std + (1 - w) * self.global_ref["std"],
            "loc_trust_score": w,
            "loc_tier_code": tier_code,
        }

    def __build_locality_stats(self, K=50):
        """Create locality map"""
        self.__compute_global_stats()
        locality_agg = self.__aggregate_by_locality()
        self.stats_map = {
            row["locality"]: self.__smooth_locality_stats(row, K)
            for _, row in locality_agg.iterrows()
        }

        save_json(self.config.root_dir / "locality_stats.json", self.stats_map)
        save_json(self.config.root_dir / "global_ref.json", self.global_ref)

    def __add_features(self, data: pd.DataFrame):
        """Add locality features to a dataframe."""

        data = self.__map_locality_class(data)
        data["class_pi"] = data["loc_class"].map(self.class_pi)
        data["loc_pi"] = data["locality"].map(self.loc_pi)

        for stat in ["loc_std_dev", "loc_trust_score", "loc_tier_code"]:
            data[stat] = data["locality"].map(
                lambda x: self.stats_map.get(x, {}).get(stat, None)
            )

        return data

    def __amenity_features(self):
        """Add some amenity features"""

        LUXURY_AMENITIES = [
            "air_conditioning",
            "chandelier",
            "microwave",
            "dishwasher",
            "refrigerator",
            "tv",
            "wi_fi",
            "hot_water",
        ]

        OTHER_AMENITIES = [
            "tiled_floor",
            "wardrobe",
            "kitchen_cabinets",
            "kitchen_shelf",
            "balcony",
            "24_hour_electricity",
            "pre_paid_meter",
            "pop_ceiling",
            "dining_area",
            "apartment",
        ]

        to_drop = LUXURY_AMENITIES + OTHER_AMENITIES

        self.train["luxury_score"] = self.train[LUXURY_AMENITIES].sum(axis=1)
        self.test["luxury_score"] = self.test[LUXURY_AMENITIES].sum(axis=1)

        self.train["amenity_count"] = self.train[LUXURY_AMENITIES].sum(
            axis=1
        ) + self.train[OTHER_AMENITIES].sum(axis=1)

        self.train = self.train.drop(
            columns=[column for column in to_drop if column in self.train.columns]
        )
        self.test = self.test.drop(
            columns=[column for column in to_drop if column in self.test.columns]
        )

    def __add_unit_density(self):
        self.train["unit_density"] = self.train["house_type"].map(self.unit_density)
        self.test["unit_density"] = self.test["house_type"].map(self.unit_density)

    def __select_columns(self):
        columns = [
            "bathrooms",
            "bedrooms",
            "condition",
            "furnishing",
            "loc",
            "lat",
            "lng",
            "luxury_score",
            "unit_density",
            "loc_class",
            "class_pi",
            "loc_pi",
            "loc_std_dev",
            "loc_trust_score",
            "loc_tier_code",
            "price",
            "log_price",
        ]

        self.train = self.train[columns]
        self.test = self.test[columns]

    def __save(self):
        self.train.to_csv(
            os.path.join(self.config.root_dir, "features_train.csv"), index=False
        )
        self.test.to_csv(
            os.path.join(self.config.root_dir, "features_test.csv"), index=False
        )

    def transform(self):
        """Apply feature engineering transform"""
        self.__create_log_price()
        self.__amenity_features()
        self.__add_unit_density()
        self.train = self.__map_locality_class(self.train)
        self.__calculate_price_indices()
        self.__build_locality_stats(K=50)
        self.train = self.__add_features(self.train)
        self.test = self.__add_features(self.test)
        self.__select_columns()
        self.__save()
