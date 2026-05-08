import os
from pathlib import Path

import joblib
import mlflow
import mlflow.catboost
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from mlflow.models.signature import infer_signature
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from core.common import create_directories, save_json
from core.logger import logger


def train_model(config: dict) -> None:
    section = config["model_trainer"]
    params = config["model"]["CatBoost"]
    create_directories([section["root_dir"]])

    train = pd.read_csv(section["train"])
    test = pd.read_csv(section["test"])

    target_col = "log_price"
    train_x = train.drop([target_col], axis=1)
    test_x = test.drop([target_col], axis=1)
    train_y = train[[target_col]]
    test_y = test[[target_col]]

    model = CatBoostRegressor(**params)
    model.fit(train_x, train_y, logging_level="Silent")

    y_pred = model.predict(test_x)
    mae = float(mean_absolute_error(test_y, y_pred))
    rmse = float(np.sqrt(mean_squared_error(test_y, y_pred)))
    r2 = float(r2_score(test_y, y_pred))

    out = os.path.join(section["root_dir"], section["model_name"])
    joblib.dump(model, out)

    logger.info(f"Model trained. Saved to {out}")
    logger.info(f"MAE={mae:.2f},  RMSE={rmse:.2f},  R²={r2:.4f}")


def evaluate_model(config: dict) -> None:
    section = config["model_evaluation"]
    params = config["model"]["CatBoost"]
    create_directories([section["root_dir"]])

    test_data = pd.read_csv(section["test"])
    model = joblib.load(section["model_path"])

    target_col = "log_price"
    test_x = test_data.drop([target_col], axis=1)
    test_y = test_data[[target_col]]

    mlflow.set_tracking_uri(
        section.get(
            "mlflow_uri",
            os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///experiments/mlflow.db"),
        )
    )
    mlflow.set_experiment("Model_Evaluation_Experiment")

    with mlflow.start_run():
        preds = model.predict(test_x)
        rmse = float(np.sqrt(mean_squared_error(test_y, preds)))
        mae = float(mean_absolute_error(test_y, preds))
        r2 = float(r2_score(test_y, preds))

        scores = {"rmse": rmse, "mae": mae, "r2": r2}
        save_json(path=Path(section["metric_file"]), data=scores)

        mlflow.log_params(params)
        mlflow.log_metrics({"rmse": rmse, "r2": r2, "mae": mae})

        signature = infer_signature(test_x, preds)
        mlflow.catboost.log_model(
            cb_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name="CatBoostRegressor",
        )
