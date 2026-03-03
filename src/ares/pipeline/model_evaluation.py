from ares.config.configuration import ConfigurationManager
from ares.components.model_evaluation import ModelEvaluation
from ares import logger

STAGE_NAME = "Model Evaluation"


class ModelEvaluationPipeline:
    def main(self) -> None:
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        evaluator = ModelEvaluation(config=model_evaluation_config)
        evaluator.log_into_mlflow()


if __name__ == "__main__":
    try:
        logger.info(f">>>> Stage {STAGE_NAME} started <<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>> Stage {STAGE_NAME} completed <<<<")
    except Exception:
        logger.exception("Stage failed: %s", STAGE_NAME)
        raise
