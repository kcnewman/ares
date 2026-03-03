import os
import zipfile
from collections.abc import Callable

from ares import logger
from ares.pipeline.data_processing import DataProcessingPipeline
from ares.pipeline.data_split import DataSplitPipeline
from ares.pipeline.data_validation import DataValidationPipeline
from ares.pipeline.feature_engineering import FeatureEngineeringPipeline
from ares.pipeline.model_evaluation import ModelEvaluationPipeline
from ares.pipeline.model_trainer import ModelTrainingPipeline


def _run_stage(stage_name: str, stage_fn: Callable[[], None]) -> None:
    logger.info(f">>>> Stage: {stage_name} started <<<<")
    try:
        stage_fn()
        logger.info(f">>>> Stage: {stage_name} completed <<<<\n\nx==========x")
    except Exception:
        logger.exception("Stage failed: %s", stage_name)
        raise


def _compress_artifacts() -> None:
    directory_to_zip = "artifacts"
    output_zip = "artifacts.zip"
    csv_allowlist = {
        os.path.join("artifacts", "data_processing", "preprocessed_train.csv")
    }

    if os.path.exists(directory_to_zip):
        with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(directory_to_zip):
                for file in files:
                    file_path = os.path.join(root, file)
                    normalized_path = os.path.normpath(file_path)
                    include_file = (not file.endswith(".csv")) or (
                        normalized_path in csv_allowlist
                    )
                    if include_file:
                        arcname = os.path.relpath(
                            file_path, os.path.join(directory_to_zip, "..")
                        )
                        zipf.write(file_path, arcname)

        logger.info(f"Artifacts compressed into {output_zip} .")
    else:
        logger.warning(f"Compression skipped: {directory_to_zip} directory not found.")


def main() -> None:
    stages: list[tuple[str, Callable[[], None]]] = [
        ("Data Validation", lambda: DataValidationPipeline().main()),
        ("Data Splitting", lambda: DataSplitPipeline().main()),
        ("Data Processing", lambda: DataProcessingPipeline().main()),
        ("Feature Engineering", lambda: FeatureEngineeringPipeline().main()),
        ("Model Training", lambda: ModelTrainingPipeline().main()),
        ("Model Evaluation", lambda: ModelEvaluationPipeline().main()),
        ("Artifact Compression", _compress_artifacts),
    ]
    for stage_name, stage_fn in stages:
        _run_stage(stage_name, stage_fn)


if __name__ == "__main__":
    main()
