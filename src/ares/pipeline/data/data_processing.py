from ares.config.configuration import ConfigurationManager
from ares.components.data_processing import DataProcessor
from ares import logger

STAGE_NAME = "Data Processing"


class DataProcessingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            data_processsing_config = config.get_data_processing_config()
            data_processing = DataProcessor(config=data_processsing_config)
            data_processing.process()
        except Exception as e:
            raise e


if __name__ == "__main__":
    try:
        logger.info(f">>>> Stage: {STAGE_NAME} started <<<<")
        obj = DataProcessingPipeline()
        obj.main()
        logger.info(f">>>> Stage: {STAGE_NAME} completed <<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
