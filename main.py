from ares import logger

from ares.pipeline.data.data_validation import DataValidationPipeline
from ares.pipeline.data.data_split import DataSplitPipeline
from ares.pipeline.data.data_processing import DataProcessingPipeline

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
