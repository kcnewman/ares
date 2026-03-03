from ares.config.configuration import ConfigurationManager
from ares.components.data_split import DataSplit
from ares import logger

STAGE_NAME = "Data Splitting"


class DataSplitPipeline:
    def main(self) -> None:
        config = ConfigurationManager()
        data_split_config = config.get_data_split_config()
        data_split = DataSplit(config=data_split_config)
        data_split.split()


if __name__ == "__main__":
    try:
        logger.info(f">>>> Stage: {STAGE_NAME} started <<<<")
        obj = DataSplitPipeline()
        obj.main()
        logger.info(f">>>> Stage: {STAGE_NAME} completed <<<<\n\nx==========x")
    except Exception:
        logger.exception("Stage failed: %s", STAGE_NAME)
        raise
