import importlib
from pathlib import Path
import joblib
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

class TestBuildWindows:
    """
    Unit tests for RandomForest.build_windows.
    """

    def test_build_windows_flattens(self, forest):
        """
        Flatten each window to one row of window x features columns.

        Args:
            forest (RandomForest): Random forest pipeline object.
        """

        # Build windows.
        windows = forest.build_windows(splits = fake_splits(features = forest.features, target = forest.target["column"]))

        # Check the number of columns.
        assert windows["train"]["X"].shape[1] == forest.window * len(forest.features)

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

    def test_tune_model_warns_when_all_f1_zero(self, forest, monkeypatch):
        """
        Warn when no setting finds any extreme day in val, because the best choice is then arbitrary.

        Args:
            forest (RandomForest): Random forest pipeline object.
            monkeypatch (pytest.MonkeyPatch): Pytest helper for temporary changes.
        """

        # Build windows.
        windows = forest.build_windows(splits = fake_splits(features = forest.features, target = forest.target["column"]))

        # Remove every extreme day from val so F1 is 0 for all settings.
        windows["val"]["y"] = np.zeros_like(windows["val"]["y"])

        # Record warnings instead of logging them.
        warnings = []

        # Replace the logger's warning method with the recorder.
        monkeypatch.setattr(forest.logger, "warning", warnings.append)

        # Tune the model.
        best = forest.tune_model(windows = windows)

        # Check the best F1 is 0.
        assert best["f1"] == 0

        # Check one warning was logged.
        assert len(warnings) == 1

class TestSaveModel:
    """
    Unit tests for RandomForest.save_model.
    """

    def test_save_model_bundle(self, forest, tmp_path):
        """
        Save the model with its parameters, validation F1, features and window.

        Args:
            forest (RandomForest): Random forest pipeline object.
            tmp_path (pathlib.Path): Temporary directory provided by pytest.
        """

        # Build windows.
        windows = forest.build_windows(splits = fake_splits(features = forest.features, target = forest.target["column"]))

        # Tune the model.
        best = forest.tune_model(windows = windows)

        # Build the output path.
        path = tmp_path / "models" / "rf.joblib"

        # Save the model.
        forest.save_model(best = best, path = str(path))

        # Load the bundle.
        bundle = joblib.load(path)

        # Check the bundle contents.
        assert sorted(bundle) == ["features", "model", "params", "val_f1", "window"]

class TestPipeline:
    """
    Unit tests for RandomForest.pipeline.
    """

    def test_pipeline_end_to_end(self, forest):
        """
        Turn split files into a saved, loadable model that predicts on flattened windows.

        Args:
            forest (RandomForest): Random forest pipeline object.
        """

        # Build the splits.
        splits = fake_splits(features = forest.features, target = forest.target["column"])

        # Store each split at its configured path.
        for name, df in splits.items():

            # Create the split folder.
            Path(forest.input[name]).parent.mkdir(parents = True, exist_ok = True)

            # Store the split.
            df.to_excel(forest.input[name], index = False)

        # Run the pipeline.
        forest.pipeline()

        # Load the saved bundle.
        bundle = joblib.load(forest.output["model"])

        # Check the model predicts on one flattened window.
        assert bundle["model"].predict(np.zeros((1, forest.window * len(forest.features)))).shape == (1,)
