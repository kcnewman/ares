from pathlib import Path
import ares.constants as const
from ares.utils.common import read_yaml, create_directories
from ares.entity.config_entity import (
    DataValidationConfig,
    DataSplitConfig,
    DataProcessingConfig,
)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath=const.CONFIG_FILE_PATH,
        params_filepath=const.PARAMS_FILE_PATH,
        schema_filepath=const.SCHEMA_FILE_PATH,
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            data_dir=config.data_dir,
            all_schema=schema,
        )
        return data_validation_config

    def get_data_split_config(self) -> DataSplitConfig:
        config = self.config.data_split

        create_directories([config.root_dir])

        data_split_config = DataSplitConfig(
            root_dir=config.root_dir, data_dir=config.data_dir
        )
        return data_split_config

    def get_data_processing_config(self) -> DataProcessingConfig:
        config = self.config.data_processing

        create_directories([config.root_dir])

        data_processing_config = DataProcessingConfig(
            root_dir=config.root_dir,
            data_dir=config.data_dir,
            train=config.train,
            test=config.test,
            geocode_cache=Path(config.geocode_cache),
        )
        return data_processing_config
