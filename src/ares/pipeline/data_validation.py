from ares.config.configuration import ConfigurationManager
from ares.components.data_validation import DataValidation
from ares import logger

STAGE_NAME = "Data Validation"


class DataValidationPipeline:
    def main(self) -> None:
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        is_valid = data_validation.validate()
        if not is_valid:
            raise RuntimeError("Data validation failed. Check status output for details.")


if __name__ == "__main__":
    try:
        logger.info(f">>>> Stage: {STAGE_NAME} started <<<<")
        obj = DataValidationPipeline()
        obj.main()
        logger.info(f">>>> Stage: {STAGE_NAME} completed <<<<\n\nx==========x")
    except Exception:
        logger.exception("Stage failed: %s", STAGE_NAME)
        raise
