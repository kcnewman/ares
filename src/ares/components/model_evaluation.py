from pathlib import Path

import joblib
import mlflow
import mlflow.catboost
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ares.entity.config_entity import ModelEvaluationConfig
from ares.utils.common import save_json


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def log_into_mlflow(self):
        test_data = pd.read_csv(self.config.test)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]

        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_experiment("Model_Evaluation_Experiment")

        with mlflow.start_run():
            preds = model.predict(test_x)
            (rmse, mae, r2) = self.eval_metrics(test_y, preds)

            scores = {"rmse": rmse, "mae": mae, "r2": r2}
            save_json(path=Path(self.config.metric_file), data=scores)

            mlflow.log_params(self.config.params)
            mlflow.log_metrics({"rmse": rmse, "r2": r2, "mae": mae})

            signature = infer_signature(test_x, preds)

            mlflow.catboost.log_model(  # type: ignore
                cb_model=model,
                artifact_path="model",
                signature=signature,
                registered_model_name="CatBoostRegressor",
            )
