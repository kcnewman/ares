import os

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from joblib import dump
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ares import logger
from ares.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        """Train baseline CatBoost and save model.

        Returns
        -------
        model: CatBoostRegressor
        metrics: dict[str, float]
        """
        train = pd.read_csv(self.config.train)
        test = pd.read_csv(self.config.test)

        train_x = train.drop([self.config.target_column], axis=1)
        test_x = test.drop([self.config.target_column], axis=1)
        train_y = train[[self.config.target_column]]
        test_y = test[[self.config.target_column]]

        params = self.config.params if self.config.params else {}
        model = CatBoostRegressor(**params)
        model.fit(train_x, train_y, logging_level="Silent")

        y_pred = model.predict(test_x)
        mae = float(mean_absolute_error(test_y, y_pred))
        rmse = float(np.sqrt(mean_squared_error(test_y, y_pred)))
        r2 = float(r2_score(test_y, y_pred))
        metrics = {"mae": mae, "rmse": rmse, "r2": r2}

        out = os.path.join(self.config.root_dir, self.config.model_name)
        dump(model, out)

        logger.info(f"Model trained. Saved to {out}")
        logger.info(f"MAE={mae:.2f},  RMSE={rmse:.2f},  R²={r2:.4f}")

        return model, metrics
