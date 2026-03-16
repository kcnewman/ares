from ares import logger
from ares.components.data_processing import DataProcessor
from ares.config.configuration import ConfigurationManager

STAGE_NAME = "Data Processing"


class DataProcessingPipeline:
    def main(self) -> None:
        config = ConfigurationManager()
        data_processsing_config = config.get_data_processing_config()
        data_processing = DataProcessor(config=data_processsing_config)
        data_processing.transform()


if __name__ == "__main__":
    try:
        logger.info(f">>>> Stage: {STAGE_NAME} started <<<<")
        obj = DataProcessingPipeline()
        obj.main()
        logger.info(f">>>> Stage: {STAGE_NAME} completed <<<<\n\nx==========x")
    except Exception:
        logger.exception("Stage failed: %s", STAGE_NAME)
        raise
