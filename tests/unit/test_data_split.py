import importlib
from pathlib import Path
import pandas as pd
import pytest

data_split = importlib.import_module("src.4_data_split")

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

class TestLoadFeatures:
    """
    Unit tests for DataSplit.load_features.
    """

    def test_sorts_by_date(self, tmp_path):
        """
        Return rows in date order even if the file is not sorted.

        Args:
            tmp_path (pathlib.Path): Temporary directory provided by pytest.
        """

        # Build the file path.
        path = tmp_path / "features.xlsx"

        # Store features in reverse date order.
        fake_features(n = 5).iloc[::-1].to_excel(path, index = False)

        # Load features.
        result = data_split.DataSplit().load_features(path = str(path))

        # Check the dates are sorted.
        assert result["Date"].is_monotonic_increasing

class TestSplitData:
    """
    Unit tests for DataSplit.split_data.
    """

    def test_split_sizes(self):
        """
        Give 70% of rows to training, 15% to validation and the rest to test.
        """

        # Split 20 rows.
        splits = data_split.DataSplit().split_data(df = fake_features(n = 20))

        # Check the sizes.
        assert [len(splits[name]) for name in ("train", "val", "test")] == [14, 3, 3]

    def test_split_sizes_avoid_float_rounding(self):
        """
        Keep the validation boundary exact for row counts where 0.70 + 0.15 would round down.
        """

        # Split 2262 rows, the prod feature dataset size.
        splits = data_split.DataSplit().split_data(df = fake_features(n = 2262))

        # Check the sizes.
        assert [len(splits[name]) for name in ("train", "val", "test")] == [1583, 339, 340]

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

class TestSummarizeSplits:
    """
    Unit tests for DataSplit.summarize_splits.
    """

    def test_returns_splits_unchanged(self):
        """
        Only log the summary, without changing the splits.
        """

        # Create the data split object.
        splitter = data_split.DataSplit()

        # Split the data.
        splits = splitter.split_data(df = fake_features(n = 20))

        # Summarize the splits.
        result = splitter.summarize_splits(splits = splits)

        # Check the splits are unchanged.
        assert result is splits

class TestStoreSplits:
    """
    Unit tests for DataSplit.store_splits.
    """

    def test_writes_three_files(self):
        """
        Create missing folders and write each split to its configured file.
        """

        # Create the data split object.
        splitter = data_split.DataSplit()

        # Split the data.
        splits = splitter.split_data(df = fake_features(n = 20))

        # Store the splits.
        splitter.store_splits(splits = splits)

        # Check each file holds its split.
        for name in ("train", "val", "test"):

            # Check the stored row count.
            assert len(pd.read_excel(splitter.output[name])) == len(splits[name])

class TestPipeline:
    """
    Unit tests for DataSplit.pipeline.
    """

    def test_pipeline_end_to_end(self):
        """
        Turn a feature file into three split files that together hold every row once, in order.
        """

        # Create the data split object.
        splitter = data_split.DataSplit()

        # Build the feature data.
        features = fake_features(n = 20)

        # Build the feature file path from the config.
        features_path = Path(splitter.input["features"])

        # Create the feature folder.
        features_path.parent.mkdir(parents = True, exist_ok = True)

        # Store the feature data.
        features.to_excel(features_path, index = False)

        # Run the pipeline.
        splitter.pipeline()

        # Read the stored splits back in order.
        stored = pd.concat([pd.read_excel(splitter.output[name]) for name in ("train", "val", "test")], ignore_index = True)

        # Check every row is stored once, in order.
        assert stored.equals(features)
