import os
from ares import logger
from ares.entity.config_entity import DataValidationConfig
import pandas as pd


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self) -> bool:
        try:
            validation_status = True

            data = pd.read_csv(self.config.data_dir)
            all_cols = list(data.columns)
            all_schema = dict(self.config.all_schema.items())

            for col in all_cols:
                if col not in all_schema:
                    validation_status = False
                    with open(self.config.STATUS_FILE, "w") as f:
                        f.write(f"Validation status: {validation_status}")
                    return validation_status

            for col in all_cols:
                if data[col].dtype != all_schema[col]:
                    validation_status = False
                    with open(self.config.STATUS_FILE, "w") as f:
                        f.write(f"Validation status: {validation_status}")
                    return validation_status

            with open(self.config.STATUS_FILE, "w") as f:
                f.write(f"Validation status: {validation_status}")

            return validation_status

        except Exception as e:
            raise e
