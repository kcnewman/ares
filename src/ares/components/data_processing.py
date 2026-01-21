import pandas as pd
import numpy as np
import googlemaps
import os

from ares import logger
from ares.utils.common import save_json, load_json
from ares.entity.config_entity import DataProcessingConfig
from dotenv import load_dotenv

load_dotenv()
maps = googlemaps.Client(key=os.getenv("GOOGLE_MAPS_KEY"))


class DataProcessor:
    def __init__(self, config: DataProcessingConfig):
        self.config = config
        self.train = pd.read_csv(self.config.train)
        self.test = pd.read_csv(self.config.test)

    def __code(self, x):
        """Geocode a location x using Google Maps API"""
        try:
            result = maps.geocode(x.lower(), region="gh")  # type: ignore
            if result:
                lat = result[0]["geometry"]["location"]["lat"]
                lng = result[0]["geometry"]["location"]["lng"]
                return (lat, lng)
        except Exception as e:
            logger.info(f"Error geocoding {x}: {e}")
        return (None, None)

    def __create_cache(self, df):
        """Save geocode data as json"""
        cache = {}
        locs = df["loc"].unique()
        for i, loc in enumerate(locs, 1):
            lat, lng = self.__code(loc)
            cache[loc] = {"lat": lat, "lng": lng}

        save_json(self.config.geocode_cache, cache)
        return cache

    def __get_lat_lng(self, location, geocodes_cache):
        """Fetch the latitude and longitude of a location"""
        location_lower = location.lower()
        if location_lower in geocodes_cache:
            result = geocodes_cache[location_lower]
            return (result["lat"], result["lng"])
        else:
            lat, lng = self.__code(location_lower)
            geocodes_cache[location_lower] = {"lat": lat, "lng": lng}
            save_json(self.config.geocode_cache, geocodes_cache)
            return (lat, lng)

    def __apply_lat_lng(self):
        try:
            geocodes_cache = load_json(self.config.geocode_cache)
        except FileNotFoundError:
            self.config.geocode_cache.parent.mkdir(parents=True, exist_ok=True)
            geocodes_cache = self.__create_cache(self.config.data_dir)

        self.train[["lat", "lng"]] = self.train["loc"].apply(
            lambda x: pd.Series(self.__get_lat_lng(x, geocodes_cache))
        )
        self.test[["lat", "lng"]] = self.test["loc"].apply(
            lambda x: pd.Series(self.__get_lat_lng(x, geocodes_cache))
        )

        missing_train = self.train.loc[self.train["lat"].isna(), "loc"].unique()
        missing_eval = self.test.loc[self.train["lat"].isna(), "loc"].unique()
        total_missing = set(missing_train) | set(missing_eval)

        logger.info(f"Still missing {len(total_missing)} lat/lng for: {total_missing}")
        return self

    def __drop_duplicates(self):
        before = self.train.shape[0] + self.test.shape[0]
        self.train = self.train.drop_duplicates(subset="url")
        self.test = self.test.drop_duplicates(subset="url")
        after = self.train.shape[0] + self.test.shape[0]

        logger.info(f"Dropped {before - after} duplicate rows")
        return self

    def __trim_price(self):
        x = np.log(self.train["price"])
        q1, q3 = np.percentile(x, [25, 75])
        iqr = q3 - q1
        l1 = q1 - 1.5 * iqr
        l2 = q3 + 1.5 * iqr

        train_outliers = self.train[
            (np.log(self.train["price"]) < l1) | (np.log(self.train["price"]) > l2)
        ]

        test_outliers = self.test[
            (np.log(self.test["price"]) < l1) | (np.log(self.test["price"]) > l2)
        ]

        logger.info(
            f"{len(train_outliers)} outliers removed in training set and {len(test_outliers)} outliers removed in evaluation set."
        )

        self.train = self.train.drop(index=train_outliers.index)
        self.test = self.test.drop(index=test_outliers.index)

        return self

    def __clean_condition(self):
        self.train.Condition = self.train.Condition.map(
            {
                "Newly-Built": "New",
                "Fairly Used": "Used",
                "Old": "Used",
                "Renovated": "Renovated",
            }
        )
        self.test.Condition = self.test.Condition.map(
            {
                "Newly-Built": "New",
                "Fairly Used": "Used",
                "Old": "Used",
                "Renovated": "Renovated",
            }
        )
        return self

    def __select_columns(self):
        drop_list = ["fetch_date", "Property Size", "locality_grouped"]
        self.train = self.train.drop(columns=drop_list)
        self.test = self.test.drop(columns=drop_list)
        features = self.train.shape[1]

        logger.info(f"{features} feature columns selected")
        return self

    def __drop_missing(self):
        self.train = self.train.dropna()
        self.test = self.test.dropna()

        return self

    def __rename_columns(self, data):
        for col in data.columns.to_list():
            data.rename(
                columns={col: col.lower().replace(" ", "_").replace("-", "_")},
                inplace=True,
            )
        return data

    def __save(self):
        self.train.to_csv(
            os.path.join(self.config.root_dir, "preprocessed_train.csv"), index=False
        )
        self.test.to_csv(
            os.path.join(self.config.root_dir, "preprocessed_eval.csv"), index=False
        )

    def transform(self):
        self.__apply_lat_lng()
        self.__drop_duplicates()
        self.__trim_price()
        self.__clean_condition()
        self.__select_columns()
        self.__drop_missing()
        self.train = self.__rename_columns(self.train)
        self.test = self.__rename_columns(self.test)
        self.__save()
