from ares import logger

from ares.pipeline.data_validation import DataValidationPipeline
from ares.pipeline.data_split import DataSplitPipeline
from ares.pipeline.data_processing import DataProcessingPipeline
from ares.pipeline.feature_engineering import FeatureEngineeringPipeline

STAGE_NAME = "Data Validation"
try:
    logger.info(f">>>> Stage: {STAGE_NAME} started <<<<")
    data_validation = DataValidationPipeline()
    data_validation.main()
    logger.info(f">>>> Stage: {STAGE_NAME} completed <<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Spliting"
try:
    logger.info(f">>>> Stage: {STAGE_NAME} started <<<<")
    data_split = DataSplitPipeline()
    data_split.main()
    logger.info(f">>>> Stage: {STAGE_NAME} completed <<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Processing"
try:
    logger.info(f">>>> Stage: {STAGE_NAME} started <<<<")
    data_processor = DataProcessingPipeline()
    data_processor.main()
    logger.info(f">>>> Stage: {STAGE_NAME} completed <<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Feature Engineering"
try:
    logger.info(f">>>> Stage: {STAGE_NAME} started <<<<")
    feature_pipeline = FeatureEngineeringPipeline()
    feature_pipeline.main()
    logger.info(f">>>> Stage: {STAGE_NAME} completed <<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e
