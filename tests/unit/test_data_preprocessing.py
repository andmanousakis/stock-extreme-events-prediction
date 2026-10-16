import importlib
import numpy as np
import pandas as pd
import pytest

data_preprocessing = importlib.import_module("src.pipeline.2_data_preprocessing")

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

class TestDailyReturns:
    """
    Unit tests for DataPreprocessing.daily_returns.
    """

    def test_daily_returns(self):
        """
        Calculate percentage change from the previous day and drop the first row.
        """

        # Build prices that rise 10% then fall 10%.
        df = fake_prices(dates = ["2015-01-02", "2015-01-05", "2015-01-06"], closes = [100.0, 110.0, 99.0])

        # Calculate daily returns.
        result = data_preprocessing.DataPreprocessing().daily_returns(df = df)

        # Check the returns.
        assert result["daily_return"].round(6).tolist() == [10.0, -10.0]

        # Check the first day was dropped.
        assert result["Date"].iloc[0] == pd.Timestamp("2015-01-05")

class TestExtremeEvents:
    """
    Unit tests for DataPreprocessing.extreme_events.
    """

    def test_extreme_events_both_directions(self):
        """
        Label moves beyond 2% in either direction as extreme, and exactly 2% as not extreme.
        """

        # Build returns above, below, inside and on the threshold.
        df = pd.DataFrame({"daily_return": [2.5, -3.0, 1.0, 2.0]})

        # Label extreme events.
        result = data_preprocessing.DataPreprocessing().extreme_events(df = df)

        # Check the labels.
        assert result["extreme_event"].tolist() == [1, 1, 0, 0]

class TestShiftTarget:
    """
    Unit tests for DataPreprocessing.shift_target.
    """

    def test_shift_target_next_day(self):
        """
        Set each day's target to the next day's event and drop the last day.
        """

        # Build events for four days.
        df = pd.DataFrame({"extreme_event": [0, 1, 0, 1]})

        # Create the data preprocessing object.
        preprocessing = data_preprocessing.DataPreprocessing()

        # Shift the target.
        result = preprocessing.shift_target(df = df)

        # Check each target is the next day's event.
        assert result[preprocessing.target["column"]].tolist() == [1, 0, 1]

        # Check the last day was dropped.
        assert len(result) == 3
