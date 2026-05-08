import json
from unittest.mock import mock_open, patch

import pytest

from core.pipeline.data import split


@pytest.mark.parametrize(
    "file_exists, file_content, should_read, expected_exception",
    [
        (False, "", False, FileNotFoundError),
        (
            True,
            json.dumps({"passed": False, "detail": "bad data"}),
            False,
            RuntimeError,
        ),
        (True, json.dumps({"passed": True, "detail": "ok"}), True, None),
    ],
)
def test_split_flow(
    valid_df,
    file_exists,
    file_content,
    should_read,
    expected_exception,
):
    config = {
        "data_split": {
            "root_dir": "output",
            "data_dir": "data.csv",
            "status_file": "status.txt",
        }
    }

    with (
        patch("os.path.exists", return_value=file_exists),
        patch("builtins.open", mock_open(read_data=file_content)),
        patch("core.pipeline.data.pd.read_csv", return_value=valid_df) as mock_read,
        patch("pandas.DataFrame.to_csv"),
        patch("core.pipeline.data.create_directories"),
    ):
        if expected_exception is None:
            split(config)
        else:
            with pytest.raises(expected_exception):
                split(config)

        if should_read:
            mock_read.assert_called_once()
        else:
            mock_read.assert_not_called()


def test_split_logic_and_grouping(valid_df):
    config = {
        "data_split": {
            "root_dir": "output",
            "data_dir": "data.csv",
            "status_file": "status.txt",
        }
    }

    with (
        patch("os.path.exists", return_value=True),
        patch(
            "builtins.open",
            mock_open(read_data=json.dumps({"passed": True, "detail": "ok"})),
        ),
        patch("core.pipeline.data.pd.read_csv", return_value=valid_df),
        patch("pandas.DataFrame.to_csv") as mock_csv,
        patch("core.pipeline.data.create_directories"),
    ):
        split(config)

        assert mock_csv.call_count == 2

        actual_eval_path = mock_csv.call_args_list[1][0][0]
        assert "eval.csv" in actual_eval_path
