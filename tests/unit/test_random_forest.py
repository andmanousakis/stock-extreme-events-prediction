import importlib
import numpy as np
import pandas as pd
import pytest

random_forest = importlib.import_module("src.5_random_forest")

def fake_splits(features: list = None, target: str = None) -> dict:
    """
    Build train, val and test splits where the target is easy to learn: tomorrow is extreme when today's return is positive.

    Args:
        features (list): Feature column names.
        target (str): Target column name.

    Returns:
        dict: 'train' (200 rows), 'val' (60 rows) and 'test' (60 rows) DataFrames.
    """

    # Make random data reproducible.
    rng = np.random.default_rng(0)

    # Build random feature values.
    df = pd.DataFrame(rng.normal(size = (320, len(features))), columns = features)

    # Add trading-day dates.
    df.insert(0, "Date", pd.bdate_range("2015-01-02", periods = len(df)))

    # Make the target depend on today's return.
    df[target] = (df["daily_return"] > 0).astype(int)

    # Return consecutive splits.
    return {"train": df.iloc[:200], "val": df.iloc[200:260], "test": df.iloc[260:]}

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
def forest():
    """
    Create a RandomForest with a small parameter grid so tests run quickly.

    Returns:
        RandomForest: Random forest pipeline object.
    """

    # Create the random forest object.
    model = random_forest.RandomForest()

    # Use a small grid.
    model.model_config["param_grid"] = {"n_estimators": [50], "max_depth": [3, None]}

    # Return the object.
    return model

class TestTuneModel:
    """
    Unit tests for RandomForest.tune_model.
    """

    def test_tune_model_learns_simple_rule(self, forest):
        """
        Reach a high validation F1 on data with an easy rule.

        Args:
            forest (RandomForest): Random forest pipeline object.
        """

        # Build windows.
        windows = forest.build_windows(splits = fake_splits(features = forest.features, target = forest.target["column"]))

        # Tune the model.
        best = forest.tune_model(windows = windows)

        # Check the rule was learned.
        assert best["f1"] > 0.8

        # Check the best parameters came from the grid.
        assert best["params"]["max_depth"] in (3, None)
