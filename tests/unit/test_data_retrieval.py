import importlib
import pandas as pd
import pytest

data_retrieval = importlib.import_module("src.pipeline.1_data_retrieval")

def fake_yahoo_data() -> pd.DataFrame:
    """
    Build a small DataFrame shaped like yfinance's output: a 'Date' index and (Price, Ticker) column headers.

    Returns:
        pd.DataFrame: Fake price data for two days.
    """

    # Create the date index.
    index = pd.DatetimeIndex(["2015-01-02", "2015-01-05"], name = "Date")

    # Create the two-level column headers in yfinance's order.
    columns = pd.MultiIndex.from_product([["Adj Close", "Close", "High", "Low", "Open", "Volume"], ["AAPL"]], names = ["Price", "Ticker"])

    # Return fake data.
    return pd.DataFrame([[24.1, 27.3, 27.8, 26.8, 27.8, 212818400], [23.4, 26.5, 27.1, 26.3, 27.0, 257142000]], index = index, columns = columns)

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

@pytest.fixture
def sleeps(monkeypatch) -> list:
    """
    Replace time.sleep with a recorder so retry tests run instantly.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest helper for temporary changes.

    Returns:
        list: Delays that would have been slept, in order.
    """

    # Store requested delays.
    delays = []

    # Record delays instead of sleeping.
    monkeypatch.setattr(data_retrieval.time, "sleep", delays.append)

    # Return recorded delays.
    return delays

def fake_download(monkeypatch, failures: int = 0, empty: bool = False) -> dict:
    """
    Replace yf.download with a fake that fails a set number of times before returning fake data.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest helper for temporary changes.
        failures (int): Number of calls that fail before one succeeds.
        empty (bool): Fail by returning an empty DataFrame instead of raising.

    Returns:
        dict: Call counter, under the 'calls' key.
    """

    # Count download calls.
    counter = {"calls": 0}

    def download(*args, **kwargs):
        """
        Fake yf.download.

        Args:
            *args (tuple): Ignored positional arguments.
            **kwargs (dict): Ignored keyword arguments.

        Returns:
            pd.DataFrame: Fake data, or an empty DataFrame for an empty failure.
        """

        # Count this call.
        counter["calls"] += 1

        # Fail while failures remain.
        if counter["calls"] <= failures:

            # Return no rows for an empty failure.
            if empty:

                # Return empty data.
                return pd.DataFrame()

            # Raise a network-style error.
            raise ConnectionError("timeout")

        # Return fake data.
        return fake_yahoo_data()

    # Replace yf.download with the fake.
    monkeypatch.setattr(data_retrieval.yf, "download", download)

    # Return call counter.
    return counter

class TestMakeRequest:
    """
    Unit tests for DataRetrieval.make_request.
    """

    def test_retries_with_backoff_then_succeeds(self, monkeypatch, sleeps):
        """
        Retry after failures with doubling delays and return data once a download works.

        Args:
            monkeypatch (pytest.MonkeyPatch): Pytest helper for temporary changes.
            sleeps (list): Recorded retry delays.
        """

        # Make the first two downloads fail.
        counter = fake_download(monkeypatch, failures = 2)

        # Request data.
        df = data_retrieval.DataRetrieval().make_request(symbol = "AAPL", start = "2015-01-01", end = "2015-01-06")

        # Check data was returned.
        assert len(df) == 2

        # Check it took three attempts.
        assert counter["calls"] == 3

        # Check the delays doubled.
        assert sleeps == [2, 4]

    def test_gives_up_after_max_retries(self, monkeypatch, sleeps):
        """
        Stop the pipeline after every attempt returns no data.

        Args:
            monkeypatch (pytest.MonkeyPatch): Pytest helper for temporary changes.
            sleeps (list): Recorded retry delays.
        """

        # Make every download return no rows.
        counter = fake_download(monkeypatch, failures = 99, empty = True)

        # Create the data retrieval object.
        retrieval = data_retrieval.DataRetrieval()

        # Get the retry configs.
        request = retrieval.request

        # Check the step stops the pipeline.
        with pytest.raises(SystemExit):

            # Request data.
            retrieval.make_request(symbol = "AAPL", start = "2015-01-01", end = "2015-01-06")

        # Check it tried max_retries times.
        assert counter["calls"] == request["max_retries"]

        # Check the delays doubled, capped at max_delay, with no wait after the last attempt.
        assert sleeps == [min(request["base_delay"] * 2 ** i, request["max_delay"]) for i in range(request["max_retries"] - 1)]

class TestApplySchema:
    """
    Unit tests for DataRetrieval.apply_schema.
    """

    def test_missing_column_stops_pipeline(self):
        """
        Stop the pipeline when a schema column is missing.
        """

        # Build data without a Volume column.
        df = pd.DataFrame({"Date": ["2015-01-02"], "Adj Close": [1.0], "Close": [1.0], "High": [1.0], "Low": [1.0], "Open": [1.0]})

        # Check the step stops the pipeline.
        with pytest.raises(SystemExit):

            # Apply the schema.
            data_retrieval.DataRetrieval().apply_schema(df = df)
