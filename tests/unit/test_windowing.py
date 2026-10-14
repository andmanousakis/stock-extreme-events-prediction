import numpy as np
import pandas as pd
from utils.windowing import Windowing

def fake_split(start: int = 0, n: int = 5) -> pd.DataFrame:
    """
    Build a small split whose feature values equal the row number, so windows are easy to check.

    Args:
        start (int): Row number of the first row.
        n (int): Number of rows.

    Returns:
        pd.DataFrame: Fake split with 'Date', two features and a target.
    """

    # Get the row numbers.
    rows = list(range(start, start + n))

    # Return fake data.
    return pd.DataFrame({
        "Date": pd.bdate_range("2015-01-02", periods = start + n)[start:],
        "a": [float(r) for r in rows],
        "b": [float(r) * 10 for r in rows],
        "target": [r % 2 for r in rows],
    })

def fake_splits() -> dict:
    """
    Build consecutive train (6 rows), val (2 rows) and test (2 rows) splits.

    Returns:
        dict: 'train', 'val' and 'test' DataFrames.
    """

    # Return consecutive splits.
    return {"train": fake_split(start = 0, n = 6), "val": fake_split(start = 6, n = 2), "test": fake_split(start = 8, n = 2)}

class TestWindowing:
    """
    Unit tests for Windowing.build.
    """

    def test_window_content_and_target(self):
        """
        Fill each window with consecutive days and take the target of its last day.
        """

        # Build windows of 3 days.
        windows = Windowing(window = 3).build(splits = fake_splits(), features = ["a", "b"], target = "target")

        # Check the first window holds rows 0, 1 and 2.
        assert np.array_equal(windows["train"]["X"][0][:, 0], [0.0, 1.0, 2.0])

        # Check its target is row 2's target.
        assert windows["train"]["y"][0] == 2 % 2

    def test_lookback_across_splits(self):
        """
        Let the first validation window look back into training, and keep every validation and test day.
        """

        # Build windows of 3 days.
        windows = Windowing(window = 3).build(splits = fake_splits(), features = ["a", "b"], target = "target")

        # Check the first validation window holds train rows 4 and 5 and val row 6.
        assert np.array_equal(windows["val"]["X"][0][:, 0], [4.0, 5.0, 6.0])

        # Check train loses window - 1 rows and val and test keep all theirs.
        assert [len(windows[name]["y"]) for name in ("train", "val", "test")] == [4, 2, 2]

    def test_targets_stay_in_split(self):
        """
        Take every validation target from the validation rows.
        """

        # Build the splits.
        splits = fake_splits()

        # Build windows of 3 days.
        windows = Windowing(window = 3).build(splits = splits, features = ["a", "b"], target = "target")

        # Check the validation targets are the validation rows' targets.
        assert windows["val"]["y"].tolist() == splits["val"]["target"].tolist()
