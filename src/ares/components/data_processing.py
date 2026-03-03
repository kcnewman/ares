import os

import googlemaps
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from ares import logger
from ares.entity.config_entity import DataProcessingConfig
from ares.utils.common import load_json, save_json

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


class DataProcessor:
    def __init__(self, config: DataProcessingConfig):
        self.config = config
        self.train = pd.read_csv(self.config.train)
        self.test = pd.read_csv(self.config.test)
        self.maps_client = _build_maps_client()

        try:
            self.geocode_cache = load_json(self.config.geocode_cache)
        except Exception:
            self.geocode_cache = {}

    # ---------- reusable functions ----------

    def clean_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standardized cleaning pipeline for any dataframe"""
        data = data.copy()

        data = self._rename_columns(data)

        drop_list = ["fetch_date", "property_size", "locality_grouped"]
        data = data.drop(
            columns=[c for c in drop_list if c in data.columns], errors="ignore"
        )

        data = self._clean_strings(data)

        return data

    def _clean_strings(self, data: pd.DataFrame):
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

    def _rename_columns(self, data: pd.DataFrame):
        new_cols = {
            col: col.lower().replace(" ", "_").replace("-", "_") for col in data.columns
        }
        return data.rename(columns=new_cols)

    def get_lat_lng(self, location: str):
        """Fetch from cache first with API as fallback."""
        if pd.isna(location):
            return (None, None)

        loc_lower = location.lower().strip()

        if loc_lower in self.geocode_cache:
            res = self.geocode_cache[loc_lower]
            return res["lat"], res["lng"]

        if self.maps_client is None:
            return (None, None)

        try:
            result = self.maps_client.geocode(loc_lower, region="gh")
            if result:
                lat = result[0]["geometry"]["location"]["lat"]
                lng = result[0]["geometry"]["location"]["lng"]

                self.geocode_cache[loc_lower] = {"lat": lat, "lng": lng}
                save_json(self.config.geocode_cache, self.geocode_cache)
                return lat, lng
        except Exception as e:
            logger.error(f"Error geocoding {location}: {e}")

        return (None, None)

    def _add_lat_lng(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        if "loc" not in data.columns:
            return data

        locations = data["loc"].dropna().astype(str).str.strip().str.lower().unique()
        loc_to_coords = {loc: self.get_lat_lng(loc) for loc in locations}

        coords = data["loc"].astype(str).str.strip().str.lower().map(loc_to_coords)
        data["lat"] = coords.str[0]
        data["lng"] = coords.str[1]
        return data

    # ---------- training only functions ----------

    def _trim_price_outliers(self, data: pd.DataFrame, l1=None, l2=None):
        """Removes outliers using calculated limits."""

        data = data[data["price"] > 0].copy()
        x = np.log(data["price"])
        if l1 is None or l2 is None:
            q1, q3 = np.percentile(x, [25, 75])
            iqr = q3 - q1
            l1 = q1 - 1.5 * iqr
            l2 = q3 + 1.5 * iqr

        return data[(x >= l1) & (x <= l2)]

    def transform(self):
        """Full Training Pipeline"""
        self.train.drop_duplicates(subset="url", inplace=True)
        self.test.drop_duplicates(subset="url", inplace=True)

        self.train = self._trim_price_outliers(self.train)
        self.test = self._trim_price_outliers(self.test)

        self.train = self.clean_dataframe(self.train)
        self.test = self.clean_dataframe(self.test)

        self.train.dropna(inplace=True)
        self.test.dropna(inplace=True)

        self.train = self._add_lat_lng(self.train)
        self.test = self._add_lat_lng(self.test)

        # Save
        self.train.to_csv(
            os.path.join(self.config.root_dir, "preprocessed_train.csv"), index=False
        )
        self.test.to_csv(
            os.path.join(self.config.root_dir, "preprocessed_eval.csv"), index=False
        )

        logger.info(
            f"Preprocessing complete. Train: {self.train.shape}, Test: {self.test.shape}"
        )
