from ares import logger

from ares.pipeline.data.data_validation import DataValidationPipeline

STAGE_NAME = "Data Validation"
try:
    logger.info(f">>>> Stage: {STAGE_NAME} started <<<<")
    data_validation = DataValidationPipeline()
    data_validation.main()
    logger.info(f">>>> Stage: {STAGE_NAME} completed <<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e
