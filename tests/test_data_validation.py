import json
from unittest.mock import mock_open, patch

import pytest

from core.pipeline.data import validate


@pytest.mark.parametrize(
    "modification, expected_error",
    [
        (lambda df: df.drop(columns=["locality"]), "Missing column: locality"),
        (lambda df: df.assign(unknown_col=0), "Unexpected column: unknown_col"),
        (
            lambda df: df.astype({"price": "int64"}),
            "Type mismatch for 'price': Expected float64, got int64",
        ),
    ],
)
def test_validation_failure_scenarios(
    mock_val_config, valid_df, modification, expected_error
):
    test_df = modification(valid_df.copy())
    config = {
        "data_validation": {
            "data_dir": "dummy.csv",
            "status_file": "status.txt",
            "root_dir": ".",
        }
    }

    with (
        patch("core.pipeline.data.pd.read_csv", return_value=test_df),
        patch("core.pipeline.data.load_schema", return_value={"COLUMNS": mock_val_config.all_schema}),
        patch("builtins.open", mock_open()) as m_file,
        patch("core.pipeline.data.create_directories"),
    ):
        assert validate(config) is False

        handle = m_file()
        written = "".join(call[0][0] for call in handle.write.call_args_list)
        status = json.loads(written)
        assert status["passed"] is False
        assert expected_error in status["detail"]


def test_validate_success(mock_val_config, valid_df):
    config = {
        "data_validation": {
            "data_dir": "dummy.csv",
            "status_file": "status.txt",
            "root_dir": ".",
        }
    }

    with (
        patch("core.pipeline.data.pd.read_csv", return_value=valid_df),
        patch("core.pipeline.data.load_schema", return_value={"COLUMNS": mock_val_config.all_schema}),
        patch("builtins.open", mock_open()) as m_file,
        patch("core.pipeline.data.create_directories"),
    ):
        assert validate(config) is True

        handle = m_file()
        written = "".join(call[0][0] for call in handle.write.call_args_list)
        status = json.loads(written)
        assert status["passed"] is True


def test_validate_critical_exception():
    config = {
        "data_validation": {
            "data_dir": "dummy.csv",
            "status_file": "status.txt",
            "root_dir": ".",
        }
    }

    with (
        patch("core.pipeline.data.pd.read_csv", side_effect=Exception("Disk Error")),
        patch("core.pipeline.data.load_schema"),
        patch("builtins.open", mock_open()) as m_file,
        patch("core.pipeline.data.create_directories"),
    ):
        with pytest.raises(Exception):
            validate(config)

        handle = m_file()
        written = "".join(call[0][0] for call in handle.write.call_args_list)
        status = json.loads(written)
        assert status["passed"] is False
        assert "Disk Error" in status["detail"]
