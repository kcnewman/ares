import os
from ares import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from ares.entity.config_entity import DataSplitConfig


class DataSplit:
    def __init__(self, config: DataSplitConfig):
        self.config = config

    def split(self):
        data = pd.read_csv(self.config.data_path)

        MIN_LISTINGS = 50
        locality_counts = data["locality"].value_counts()
        rare_localitys = locality_counts[locality_counts < MIN_LISTINGS].index

        data["locality_grouped"] = data["locality"].where(
            ~data["locality"].isin(rare_localitys), other="OTHER"
        )

        train, eval = train_test_split(
            data, test_size=0.2, random_state=2025, stratify=data["locality_grouped"]
        )

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        eval.to_csv(os.path.join(self.config.root_dir, "eval.csv"), index=False)

        logger.info("Splited data into training and evaluation sets")
        logger.info(train.shape)
        logger.info(eval.shape)
