import importlib
from pathlib import Path
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

    def test_shift_target_uses_config_shift(self):
        """
        Use the configured shift to look further ahead.
        """

        # Build events for four days.
        df = pd.DataFrame({"extreme_event": [0, 1, 0, 1]})

        # Create the feature engineering object.
        engineering = feature_engineering.FeatureEngineering()

        # Look two days ahead.
        engineering.target["shift"] = 2

        # Shift the target.
        result = engineering.shift_target(df = df)

        # Check each target is the event two days later.
        assert result[engineering.target["column"]].tolist() == [0, 1]

class TestSelectColumns:
    """
    Unit tests for FeatureEngineering.select_columns.
    """

    def test_select_columns(self):
        """
        Keep the date, the feature columns and the target, and leave out today's event.
        """

        # Create the feature engineering object.
        engineering = feature_engineering.FeatureEngineering()

        # Build data with features, today's event and the target.
        df = fake_clean(prices = [1.0, 2.0]).assign(daily_return = 0.0, extreme_event = 0)

        # Add the target column.
        df[engineering.target["column"]] = 0

        # Select columns.
        result = engineering.select_columns(df = df)

        # Check the columns.
        assert list(result.columns) == ["Date"] + engineering.features + [engineering.target["column"]]

class TestStoreFeatures:
    """
    Unit tests for FeatureEngineering.store_features.
    """

    def test_writes_excel_file(self, tmp_path):
        """
        Create missing folders and write the feature dataset to Excel.

        Args:
            tmp_path (pathlib.Path): Temporary directory provided by pytest.
        """

        # Build the output path in a folder that doesn't exist yet.
        path = tmp_path / "data" / "processed" / "features.xlsx"

        # Store data.
        feature_engineering.FeatureEngineering().store_features(df = fake_clean(prices = [1.0]), path = str(path))

        # Check the file was written.
        assert pd.read_excel(path)["Close"].iloc[0] == 1.0

class TestPipeline:
    """
    Unit tests for FeatureEngineering.pipeline.
    """

    def test_pipeline_end_to_end(self):
        """
        Turn a clean file into a feature file with the right columns, two fewer rows and next-day targets.
        """

        # Create the feature engineering object.
        engineering = feature_engineering.FeatureEngineering()

        # Build prices with returns of +10%, -10%, +1% and +5%.
        clean = fake_clean(prices = [100.0, 110.0, 99.0, 99.99, 104.9895])

        # Build the clean data path from the config.
        clean_path = Path(engineering.input["clean_data"])

        # Create the clean data folder.
        clean_path.parent.mkdir(parents = True, exist_ok = True)

        # Store the clean data.
        clean.to_excel(clean_path, index = False)

        # Run the pipeline.
        engineering.pipeline()

        # Read the feature dataset.
        features = pd.read_excel(engineering.output["features"])

        # Check the columns.
        assert list(features.columns) == ["Date"] + engineering.features + [engineering.target["column"]]

        # Check the first and last days were dropped.
        assert len(features) == len(clean) - 2

        # Check each target is the next day's event: -10% yes, +1% no, +5% yes.
        assert features[engineering.target["column"]].tolist() == [1, 0, 1]
