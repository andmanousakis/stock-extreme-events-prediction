import importlib
import numpy as np
import pandas as pd
import pytest

data_preprocessing = importlib.import_module("src.2_data_preprocessing")

def fake_prices(dates: list = None, closes: list = None) -> pd.DataFrame:
    """
    Build a small price table with every raw data column, using the same value for all price columns of a day.

    Args:
        dates (list): Dates as YYYY-MM-DD strings.
        closes (list): One price per date; np.nan marks a missing value.

    Returns:
        pd.DataFrame: Fake price data.
    """

    # Return fake data.
    return pd.DataFrame({
        "Date": pd.to_datetime(dates),
        "Adj Close": closes,
        "Close": closes,
        "High": closes,
        "Low": closes,
        "Open": closes,
        "Volume": [1000] * len(dates),
    })

@pytest.fixture(autouse = True)
def run_in_tmp_path(tmp_path, monkeypatch):
    """
    Run every test from a temporary folder so log files and outputs don't land in the project.

    Args:
        tmp_path (pathlib.Path): Temporary directory provided by pytest.
        monkeypatch (pytest.MonkeyPatch): Pytest helper for temporary changes.
    """

    # Change the working directory to the temporary folder.
    monkeypatch.chdir(tmp_path)

class TestSortAndDeduplicate:
    """
    Unit tests for DataPreprocessing.sort_and_deduplicate.
    """

    def test_drops_duplicate_dates_keeping_last(self):
        """
        Keep one row per date, the last one.
        """

        # Build data with a repeated date.
        df = fake_prices(dates = ["2015-01-02", "2015-01-05", "2015-01-05"], closes = [1.0, 2.0, 9.0])

        # Sort and deduplicate.
        result = data_preprocessing.DataPreprocessing().sort_and_deduplicate(df = df)

        # Check one row is left per date.
        assert len(result) == 2

        # Check the last duplicate was kept.
        assert result["Close"].iloc[-1] == 9.0

class TestHandleMissingValues:
    """
    Unit tests for DataPreprocessing.handle_missing_values.
    """

    def test_forward_fills_gap(self):
        """
        Fill a missing value with the previous trading day's value.
        """

        # Build data with a missing price on the middle day.
        df = fake_prices(dates = ["2015-01-02", "2015-01-05", "2015-01-06"], closes = [1.0, np.nan, 3.0])

        # Handle missing values.
        result = data_preprocessing.DataPreprocessing().handle_missing_values(df = df)

        # Check the gap was filled with the previous day's value.
        assert result["Close"].tolist() == [1.0, 1.0, 3.0]

    def test_does_not_add_calendar_days(self):
        """
        Keep trading days only, without adding weekend rows.
        """

        # Build data for a Friday and the following Monday.
        df = fake_prices(dates = ["2015-01-02", "2015-01-05"], closes = [1.0, 2.0])

        # Handle missing values.
        result = data_preprocessing.DataPreprocessing().handle_missing_values(df = df)

        # Check no weekend rows were added.
        assert len(result) == 2
