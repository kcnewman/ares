from ares.config.configuration import ConfigurationManager
from ares.components.data_split import DataSplit
from ares import logger
from pathlib import Path


STAGE_NAME = "Data Spliting"


class DataSplitPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            with open(Path("artifacts/pipeline/data_validation/status.txt"), "r") as f:
                status = f.read().split(" ")[-1]

            if status == "True":
                config = ConfigurationManager()
                data_split_config = config.get_data_split_config()
                data_split = DataSplit(config=data_split_config)
                data_split.split()

            else:
                raise Exception("Data schema is not valid")
        except Exception as e:
            print(e)


if __name__ == "__main__":
    try:
        logger.info(f">>>> Stage: {STAGE_NAME} started <<<<")
        obj = DataSplitPipeline()
        obj.main()
        logger.info(f">>>> Stage: {STAGE_NAME} completed <<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
