from ares.config.configuration import ConfigurationManager
from ares.components.feature_engineering import EngineerFeatures
from ares import logger

STAGE_NAME = "Feature Engineering"


class FeatureEngineeringPipeline:
    def main(self) -> None:
        config = ConfigurationManager()
        feature_engineering_config = config.get_feature_engineering_config()
        feature_engineering = EngineerFeatures(config=feature_engineering_config)
        feature_engineering.transform()


if __name__ == "__main__":
    try:
        logger.info(f">>>> Stage: {STAGE_NAME} started <<<<")
        obj = FeatureEngineeringPipeline()
        obj.main()
        logger.info(f">>>> Stage: {STAGE_NAME} completed <<<<\n\nx==========x")
    except Exception:
        logger.exception("Stage failed: %s", STAGE_NAME)
        raise
