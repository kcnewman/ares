import os
from box.exceptions import BoxValueError
import yaml
from ares import logger
import json
import joblib
from beartype import beartype
from box import ConfigBox
from pathlib import Path
from typing import Any


@beartype
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file

    Args:
        path_to_yaml (Path): path

    Raises:
        VakueError: if yaml file is empty

    Returns:
        ConfigBox: Configbox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e


@beartype
def create_directories(path_to_directories: list, verbose=True):
    """create a list of directories

    Args:
        path_to_directories (list): list of paths
        verbose (bool, optional): logger
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")


@beartype
def save_json(path: Path, data: dict):
    """save json file

    Args:
        path (Path): path to save file
        data (dict): data to be saved
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"JSON file saved at; {path}")


@beartype
def load_json(path: Path) -> ConfigBox:
    """load json files

    Args:
        path (Path): Path to json file

    Returns:
        ConfigBox: Data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)
    logger.info(f"JSON file loaded successfully from: {path}")
    return ConfigBox(content)


@beartype
def save_bins(data: Any, path: Path):
    """Save binary file

    Args:
        data (Any): data to be saved
        path (Path): path to save
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"Binary file saved ar: {path}")


@beartype
def load_bin(path: Path) -> Any:
    """Load binary data

    Args:
        path (Path): Path to file

    Returns:
        Any: Object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"Binary file is loaded from: {path}")
    return data


@beartype
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): Path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"
