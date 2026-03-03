from ares.config.configuration import ConfigurationManager
from ares.components.data_split import DataSplit
from ares import logger

STAGE_NAME = "Data Spliting"


class DataSplitPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            data_split_config = config.get_data_split_config()
            data_split = DataSplit(config=data_split_config)
            data_split.split()
        except Exception as e:
            raise e


if __name__ == "__main__":
    try:
        logger.info(f">>>> Stage: {STAGE_NAME} started <<<<")
        obj = DataSplitPipeline()
        obj.main()
        logger.info(f">>>> Stage: {STAGE_NAME} completed <<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
