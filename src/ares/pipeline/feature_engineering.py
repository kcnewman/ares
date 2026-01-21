from ares.config.configuration import ConfigurationManager
from ares.components.feature_engineering import EngineerFeatures
from ares import logger

STAGE_NAME = "Feature Engineering"


class FeatureEngineeringPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            feature_engineering_config = config.get_feature_engineering_config()
            feature_engineering = EngineerFeatures(config=feature_engineering_config)
            feature_engineering.transform()
        except Exception as e:
            raise e


if __name__ == "__main__":
    try:
        logger.info(f">>>> Stage: {STAGE_NAME} started <<<<")
        obj = FeatureEngineeringPipeline()
        obj.main()
        logger.info(f">>>> Stage: {STAGE_NAME} completed <<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
