from unittest.mock import mock_open, patch

import pytest

from ares.components.data_validation import DataValidation


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

    with (
        patch("pandas.read_csv", return_value=test_df),
        patch("builtins.open", mock_open()) as m_file,
    ):
        validator = DataValidation(mock_val_config)
        assert validator.validate() is False

        # Verify the specific error message was written to the status file
        handle = m_file()
        handle.write.assert_any_call(f"Details: {expected_error}")


def test_validate_success(mock_val_config, valid_df):
    with (
        patch("pandas.read_csv", return_value=valid_df),
        patch("builtins.open", mock_open()) as m_file,
    ):
        validator = DataValidation(mock_val_config)
        assert validator.validate() is True

        handle = m_file()
        handle.write.assert_any_call("Validation status: True\n")


def test_validate_critical_exception(mock_val_config):
    with (
        patch("pandas.read_csv", side_effect=Exception("Disk Error")),
        patch("builtins.open", mock_open()) as m_file,
    ):
        validator = DataValidation(mock_val_config)
        with pytest.raises(Exception):
            validator.validate()

        handle = m_file()
        handle.write.assert_any_call("Validation status: False\n")
