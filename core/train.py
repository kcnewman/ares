import os
import zipfile
from collections.abc import Callable

from core.config import load_config
from core.logger import logger
from core.pipeline.data import process, split, validate
from core.pipeline.features import fit_features
from core.pipeline.model import evaluate_model, train_model


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
                        arcname = os.path.relpath(file_path, os.path.join(directory_to_zip, ".."))
                        zipf.write(file_path, arcname)

        logger.info(f"Artifacts compressed into {output_zip} .")
    else:
        logger.warning(f"Compression skipped: {directory_to_zip} directory not found.")


def main() -> None:
    config = load_config()

    stages: list[tuple[str, Callable[[], None]]] = [
        ("Data Validation", lambda: validate(config)),
        ("Data Splitting", lambda: split(config)),
        ("Data Processing", lambda: process(config)),
        ("Feature Engineering", lambda: fit_features(config)),
        ("Model Training", lambda: train_model(config)),
        ("Model Evaluation", lambda: evaluate_model(config)),
        ("Artifact Compression", _compress_artifacts),
    ]
    for stage_name, stage_fn in stages:
        _run_stage(stage_name, stage_fn)


if __name__ == "__main__":
    main()
