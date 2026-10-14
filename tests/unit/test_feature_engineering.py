import importlib
import pandas as pd
import pytest

feature_engineering = importlib.import_module("src.3_feature_engineering")

def fake_clean(prices: list = None) -> pd.DataFrame:
    """
    Build a small clean price table with one row per trading day, using the same value for all price columns of a day.

    Args:
        prices (list): One Adj Close price per day.

    Returns:
        pd.DataFrame: Fake clean data.
    """

    # Return fake data.
    return pd.DataFrame({
        "Date": pd.bdate_range("2015-01-02", periods = len(prices)),
        "Adj Close": prices,
        "Close": prices,
        "High": prices,
        "Low": prices,
        "Open": prices,
        "Volume": [1000] * len(prices),
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

class TestDailyReturns:
    """
    Unit tests for FeatureEngineering.daily_returns.
    """

    def test_daily_returns(self):
        """
        Calculate percentage change from the previous day and drop the first row.
        """

        # Build prices that rise 10% then fall 10%.
        df = fake_clean(prices = [100.0, 110.0, 99.0])

        # Calculate daily returns.
        result = feature_engineering.FeatureEngineering().daily_returns(df = df)

        # Check the returns.
        assert result["daily_return"].round(6).tolist() == [10.0, -10.0]

        # Check the first day was dropped.
        assert result["Date"].iloc[0] == pd.Timestamp("2015-01-05")

class TestExtremeEvents:
    """
    Unit tests for FeatureEngineering.extreme_events.
    """

    def test_extreme_events_both_directions(self):
        """
        Label moves beyond 2% in either direction as extreme, and exactly 2% as not extreme.
        """

        # Build returns above, below, inside and on the threshold.
        df = pd.DataFrame({"daily_return": [2.5, -3.0, 1.0, 2.0]})

        # Label extreme events.
        result = feature_engineering.FeatureEngineering().extreme_events(df = df)

        # Check the labels.
        assert result["extreme_event"].tolist() == [1, 1, 0, 0]

class TestShiftTarget:
    """
    Unit tests for FeatureEngineering.shift_target.
    """

    def test_shift_target_next_day(self):
        """
        Set each day's target to the next day's event and drop the last day.
        """

        # Build events for four days.
        df = pd.DataFrame({"extreme_event": [0, 1, 0, 1]})

        # Create the feature engineering object.
        engineering = feature_engineering.FeatureEngineering()

        # Shift the target.
        result = engineering.shift_target(df = df)

        # Check each target is the next day's event.
        assert result[engineering.target["column"]].tolist() == [1, 0, 1]

        # Check the last day was dropped.
        assert len(result) == 3
