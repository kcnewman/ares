from ares.entity.config_entity import DataValidationConfig
import pandas as pd


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate(self) -> bool:
        try:
            validation_status = True
            error_msg = "All checks passed"

            data = pd.read_csv(self.config.data_dir)
            all_cols = list(data.columns)
            all_schema = dict(self.config.all_schema.items())

            for col in all_schema.keys():
                if col not in all_cols:
                    validation_status = False
                    error_msg = f"Missing column: {col}"
                    break

            if validation_status:
                for col in all_cols:
                    if col not in all_schema:
                        validation_status = False
                        error_msg = f"Unexpected column: {col}"
                        break
            if validation_status:
                for col in all_cols:
                    actual_dtype = str(data[col].dtype)
                    expected_dtype = all_schema[col]
                    if actual_dtype != expected_dtype:
                        validation_status = False
                        validation_status = False
                        error_msg = f"Type mismatch for '{col}': Expected {expected_dtype}, got {actual_dtype}"
                        break

            with open(self.config.STATUS_FILE, "w") as f:
                f.write(f"Validation status: {validation_status}\n")
                f.write(f"Details: {error_msg}")

            return validation_status

        except Exception as e:
            with open(self.config.STATUS_FILE, "w") as f:
                f.write("Validation status: False\n")
                f.write(f"Details: Exception occurred: {str(e)}")
            raise e
