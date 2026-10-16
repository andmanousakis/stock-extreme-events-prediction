import importlib
import pandas as pd
import pytest

data_split = importlib.import_module("src.pipeline.4_data_split")

def fake_features(n: int = 20) -> pd.DataFrame:
    """
    Build a small feature dataset with one row per trading day and an alternating target.

    Args:
        n (int): Number of rows.

    Returns:
        pd.DataFrame: Fake feature data.
    """

    # Return fake data.
    return pd.DataFrame({
        "Date": pd.bdate_range("2015-01-02", periods = n),
        "Close": [i + 0.5 for i in range(n)],
        "extreme_event_next_day": [i % 2 for i in range(n)],
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

class TestSplitData:
    """
    Unit tests for DataSplit.split_data.
    """

    def test_splits_in_order_without_overlap(self):
        """
        Keep the oldest rows in training and the newest in test, with no shared or lost rows.
        """

        # Build the feature data.
        df = fake_features(n = 20)

        # Split the data.
        splits = data_split.DataSplit().split_data(df = df)

        # Check training ends before validation starts.
        assert splits["train"]["Date"].max() < splits["val"]["Date"].min()

        # Check validation ends before test starts.
        assert splits["val"]["Date"].max() < splits["test"]["Date"].min()

        # Check the splits put back together equal the input.
        assert pd.concat([splits["train"], splits["val"], splits["test"]], ignore_index = True).equals(df)
