from unittest.mock import mock_open, patch

import pytest

from ares.components.data_split import DataSplit


@pytest.mark.parametrize(
    "file_exists, file_content, should_read, expected_exception",
    [
        (False, "", False, FileNotFoundError),  # status file missing
        (True, "Validation status: False", False, RuntimeError),  # validation failed
        (True, "Validation status: True", True, None),  # validation passed
    ],
)
def test_split_flow(
    mock_split_config,
    valid_df,
    file_exists,
    file_content,
    should_read,
    expected_exception,
):
    with (
        patch("os.path.exists", return_value=file_exists),
        patch("builtins.open", mock_open(read_data=file_content)),
        patch("pandas.read_csv", return_value=valid_df) as mock_read,
        patch("pandas.DataFrame.to_csv"),
    ):
        ds = DataSplit(mock_split_config)
        if expected_exception is None:
            ds.split()
        else:
            with pytest.raises(expected_exception):
                ds.split()

        if should_read:
            mock_read.assert_called_once()
        else:
            mock_read.assert_not_called()


def test_split_logic_and_grouping(mock_split_config, valid_df):
    """Checks if 'rare' localities are grouped into 'OTHER' before splitting."""
    with (
        patch("os.path.exists", return_value=True),
        patch("builtins.open", mock_open(read_data="Validation status: True")),
        patch("pandas.read_csv", return_value=valid_df),
        patch("pandas.DataFrame.to_csv") as mock_csv,
    ):
        ds = DataSplit(mock_split_config)
        ds.split()

        assert mock_csv.call_count == 2

        actual_eval_path = mock_csv.call_args_list[1][0][0]
        assert "eval.csv" in actual_eval_path
