import os

import pandas as pd
from sklearn.model_selection import train_test_split

from ares import logger
from ares.entity.config_entity import DataSplitConfig


class DataSplit:
    def __init__(self, config: DataSplitConfig):
        self.config = config

    def split(self) -> None:
        status_dir = self.config.status_file

        if not os.path.exists(status_dir):
            message = (
                f"Validation status file missing at {status_dir}. "
                "Cannot continue data split."
            )
            logger.error(message)
            raise FileNotFoundError(message)

        with open(status_dir, "r") as f:
            first_line = f.readline()
            if "Validation status: True" not in first_line:
                message = (
                    "Data Validation failed. "
                    "See status file for details before splitting."
                )
                logger.error(message)
                raise RuntimeError(message)

        logger.info("Validation passed. Starting data split...")

        data = pd.read_csv(self.config.data_dir)

        min_listings = 50
        locality_counts = data["locality"].value_counts()
        rare_localities = locality_counts[locality_counts < min_listings].index

        data["locality_grouped"] = data["locality"].where(
            ~data["locality"].isin(rare_localities), other="OTHER"
        )

        train, eval = train_test_split(
            data, test_size=0.2, random_state=2025, stratify=data["locality_grouped"]
        )

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        eval.to_csv(os.path.join(self.config.root_dir, "eval.csv"), index=False)

        logger.info("Split data into training and evaluation sets")
        logger.info(f"Train shape: {train.shape}")
        logger.info(f"Test shape: {eval.shape}")
