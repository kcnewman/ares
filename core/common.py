import json
import os
from pathlib import Path

from core.logger import logger


def create_directories(paths: list[str | Path]) -> None:
    for path in paths:
        os.makedirs(path, exist_ok=True)
        logger.info(f"Created directory at: {path}")


def save_json(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"JSON file saved at: {path}")


def load_json(path: Path) -> dict:
    with open(path) as f:
        content = json.load(f)
    logger.info(f"JSON file loaded successfully from: {path}")
    return content
