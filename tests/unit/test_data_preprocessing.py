import importlib
from pathlib import Path
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

    def test_sorts_by_date(self):
        """
        Return rows in date order.
        """

        # Build data with dates out of order.
        df = fake_prices(dates = ["2015-01-06", "2015-01-02", "2015-01-05"], closes = [3.0, 1.0, 2.0])

        # Sort and deduplicate.
        result = data_preprocessing.DataPreprocessing().sort_and_deduplicate(df = df)

        # Check the dates are sorted.
        assert list(result["Date"].dt.strftime("%Y-%m-%d")) == ["2015-01-02", "2015-01-05", "2015-01-06"]

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

    def test_drops_leading_unfillable_row(self):
        """
        Drop a first-day row that has no previous day to fill from.
        """

        # Build data with a missing price on the first day.
        df = fake_prices(dates = ["2015-01-02", "2015-01-05", "2015-01-06"], closes = [np.nan, 2.0, 3.0])

        # Handle missing values.
        result = data_preprocessing.DataPreprocessing().handle_missing_values(df = df)

        # Check the first row was dropped.
        assert result["Close"].tolist() == [2.0, 3.0]

        # Check nothing is missing.
        assert not result.isna().any().any()

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

class TestApplySchema:
    """
    Unit tests for DataPreprocessing.apply_schema.
    """

    def test_missing_column_stops_pipeline(self):
        """
        Stop the pipeline when a schema column is missing.
        """

        # Build data without a Volume column.
        df = fake_prices(dates = ["2015-01-02"], closes = [1.0]).drop(columns = ["Volume"])

        # Check the step stops the pipeline.
        with pytest.raises(SystemExit):

            # Apply the schema.
            data_preprocessing.DataPreprocessing().apply_schema(df = df)

class TestStoreCleanData:
    """
    Unit tests for DataPreprocessing.store_clean_data.
    """

    def test_writes_excel_file(self, tmp_path):
        """
        Create missing folders and write the clean data to Excel.

        Args:
            tmp_path (pathlib.Path): Temporary directory provided by pytest.
        """

        # Build the output path in a folder that doesn't exist yet.
        path = tmp_path / "data" / "interim" / "clean.xlsx"

        # Build data to store.
        df = fake_prices(dates = ["2015-01-02"], closes = [1.0])

        # Store data.
        data_preprocessing.DataPreprocessing().store_clean_data(df = df, path = str(path))

        # Read the stored file.
        stored = pd.read_excel(path)

        # Check the stored value.
        assert stored["Close"].iloc[0] == 1.0

class TestPipeline:
    """
    Unit tests for DataPreprocessing.pipeline.
    """

    def test_pipeline_cleans_messy_raw_file(self):
        """
        Clean a raw file with unsorted dates, a duplicate date and a gap, and store sorted, unique, complete data.
        """

        # Create the data preprocessing object.
        preprocessing = data_preprocessing.DataPreprocessing()

        # Build messy raw data: unsorted, a duplicate date and a missing price.
        raw = fake_prices(dates = ["2015-01-06", "2015-01-02", "2015-01-05", "2015-01-05"], closes = [3.0, 1.0, np.nan, 2.0])

        # Build the raw data path from the config.
        raw_path = Path(preprocessing.input["raw_data"])

        # Create the raw data folder.
        raw_path.parent.mkdir(parents = True, exist_ok = True)

        # Store the messy raw data.
        raw.to_excel(raw_path, index = False)

        # Run the pipeline.
        preprocessing.pipeline()

        # Read the clean data.
        clean = pd.read_excel(preprocessing.output["clean_data"])

        # Check one sorted row per trading day.
        assert list(clean["Date"].dt.strftime("%Y-%m-%d")) == ["2015-01-02", "2015-01-05", "2015-01-06"]

        # Check the duplicate kept its last, complete row.
        assert clean["Close"].tolist() == [1.0, 2.0, 3.0]

        # Check nothing is missing.
        assert not clean.isna().any().any()
