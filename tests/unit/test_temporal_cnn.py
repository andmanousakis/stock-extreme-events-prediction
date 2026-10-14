import importlib
import joblib
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from src.networks.temporal_cnn import TemporalCNNNetwork

temporal_cnn = importlib.import_module("src.6_temporal_cnn")

def fake_splits(features: list = None, target: str = None) -> dict:
    """
    Build train, val and test splits where the target is easy to learn: tomorrow is extreme when today's return is positive.

    Args:
        features (list): Feature column names.
        target (str): Target column name.

    Returns:
        dict: 'train' (300 rows), 'val' (80 rows) and 'test' (80 rows) DataFrames.
    """

    # Make random data reproducible.
    rng = np.random.default_rng(0)

    # Build random feature values.
    df = pd.DataFrame(rng.normal(size = (460, len(features))), columns = features)

    # Add trading-day dates.
    df.insert(0, "Date", pd.bdate_range("2015-01-02", periods = len(df)))

    # Make the target depend on today's return.
    df[target] = (df["daily_return"] > 0).astype(int)

    # Return consecutive splits.
    return {"train": df.iloc[:300].reset_index(drop = True), "val": df.iloc[300:380].reset_index(drop = True), "test": df.iloc[380:].reset_index(drop = True)}

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
def cnn():
    """
    Create a TemporalCNN with short training so tests run quickly.

    Returns:
        TemporalCNN: Temporal CNN pipeline object.
    """

    # Create the temporal CNN object.
    model = temporal_cnn.TemporalCNN()

    # Train for at most 40 epochs.
    model.training["max_epochs"] = 40

    # Stop after 5 epochs without improvement.
    model.training["patience"] = 5

    # Return the object.
    return model

def prepared_windows(cnn) -> dict:
    """
    Scale the fake splits and build their windows.

    Args:
        cnn (TemporalCNN): Temporal CNN pipeline object.

    Returns:
        dict: Windows for 'train', 'val' and 'test'.
    """

    # Scale the fake splits and build windows.
    return cnn.build_windows(splits = cnn.scale_features(splits = fake_splits(features = cnn.features, target = cnn.target["column"])))

class TestScaleFeatures:
    """
    Unit tests for TemporalCNN.scale_features.
    """

    def test_scaling_uses_train_only(self, cnn):
        """
        Scale train to mean 0 and std 1, and scale val with train's statistics rather than its own.

        Args:
            cnn (TemporalCNN): Temporal CNN pipeline object.
        """

        # Build the splits.
        splits = fake_splits(features = cnn.features, target = cnn.target["column"])

        # Shift val so its own mean is far from train's.
        splits["val"][cnn.features] = splits["val"][cnn.features] + 10

        # Scale the features.
        scaled = cnn.scale_features(splits = splits)

        # Check train has mean 0.
        assert np.allclose(scaled["train"][cnn.features].mean(), 0, atol = 1e-6)

        # Check train has standard deviation 1.
        assert np.allclose(scaled["train"][cnn.features].std(ddof = 0), 1, atol = 1e-6)

        # Check val keeps its shift, so it was scaled with train's statistics.
        assert (scaled["val"][cnn.features].mean() > 5).all()

class TestTrainModel:
    """
    Unit tests for TemporalCNN.train_model.
    """

    def test_learns_simple_rule(self, cnn):
        """
        Reach a high validation F1 on data with an easy rule.

        Args:
            cnn (TemporalCNN): Temporal CNN pipeline object.
        """

        # Build windows.
        windows = prepared_windows(cnn = cnn)

        # Train the network, pick the threshold and score it.
        result = cnn.score_model(result = cnn.tune_threshold(result = cnn.train_model(network = cnn.build_network(), windows = windows), windows = windows), windows = windows)

        # Check the rule was learned.
        assert result["metrics"]["val"]["f1"] > 0.8

    def test_early_stopping_restores_best(self, cnn):
        """
        Stop patience epochs after the best epoch and restore that epoch's weights.

        Args:
            cnn (TemporalCNN): Temporal CNN pipeline object.
        """

        # Build windows.
        windows = prepared_windows(cnn = cnn)

        # Train the network.
        result = cnn.train_model(network = cnn.build_network(), windows = windows)

        # Check it stopped patience epochs after the best, or at the epoch limit.
        assert result["epochs_run"] in (result["best_epoch"] + cnn.training["patience"], cnn.training["max_epochs"])

        # Switch off dropout.
        result["network"].eval()

        # Recalculate the validation loss of the returned network.
        with torch.no_grad():

            # Score the validation windows.
            val_loss = nn.CrossEntropyLoss()(result["network"](torch.tensor(windows["val"]["X"])), torch.tensor(windows["val"]["y"], dtype = torch.long)).item()

        # Check the returned network has the best epoch's validation loss.
        assert val_loss == pytest.approx(result["best_val_loss"], rel = 1e-5)

class TestSaveModel:
    """
    Unit tests for TemporalCNN.save_model.
    """

    def test_save_and_rebuild(self, cnn, tmp_path):
        """
        Save a bundle that rebuilds a network predicting exactly like the trained one.

        Args:
            cnn (TemporalCNN): Temporal CNN pipeline object.
            tmp_path (pathlib.Path): Temporary directory provided by pytest.
        """

        # Build windows.
        windows = prepared_windows(cnn = cnn)

        # Train the network, pick the threshold and score it.
        result = cnn.score_model(result = cnn.tune_threshold(result = cnn.train_model(network = cnn.build_network(), windows = windows), windows = windows), windows = windows)

        # Build the output path.
        path = tmp_path / "models" / "tcnn.joblib"

        # Save the model.
        cnn.save_model(result = result, path = str(path))

        # Load the bundle.
        bundle = joblib.load(path)

        # Rebuild the network from the bundle.
        rebuilt = TemporalCNNNetwork(num_features = len(bundle["features"]), window = bundle["window"], **bundle["network"])

        # Load the saved weights.
        rebuilt.load_state_dict(bundle["state_dict"])

        # Get the test windows.
        x = torch.tensor(windows["test"]["X"])

        # Check the rebuilt network predicts exactly like the trained one.
        assert torch.equal(rebuilt.predict_proba(x), result["network"].predict_proba(x))
