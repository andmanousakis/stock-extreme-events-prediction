import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from src.models.temporal_cnn import TemporalCNNModel

FEATURES = ["Open", "High", "Low", "Close", "Volume", "daily_return"]

TARGET = "extreme_event_next_day"

def fake_splits() -> dict:
    """
    Build train, val and test splits where the target is easy to learn: tomorrow is extreme when today's return is positive.

    Returns:
        dict: 'train' (300 rows), 'val' (80 rows) and 'test' (80 rows) DataFrames.
    """

    # Make random data reproducible.
    rng = np.random.default_rng(0)

    # Build random feature values.
    df = pd.DataFrame(rng.normal(size = (460, len(FEATURES))), columns = FEATURES)

    # Add trading-day dates.
    df.insert(0, "Date", pd.bdate_range("2015-01-02", periods = len(df)))

    # Make the target depend on today's return.
    df[TARGET] = (df["daily_return"] > 0).astype(int)

    # Return consecutive splits.
    return {"train": df.iloc[:300].reset_index(drop = True), "val": df.iloc[300:380].reset_index(drop = True), "test": df.iloc[380:].reset_index(drop = True)}

def val_f1(model) -> float:
    """
    Fit a model on the fake splits and score it on val.

    Args:
        model (TemporalCNNModel): Untrained model.

    Returns:
        float: Validation F1.
    """

    # Fit and predict.
    predictions = model.fit(splits = fake_splits()).predict(splits = fake_splits())

    # Score val.
    return f1_score(predictions["val"]["y"], predictions["val"]["pred"])

class TestTemporalCNNModel:
    """
    Unit tests for TemporalCNNModel.
    """

    def test_scaling_uses_train_only(self):
        """
        Fit the scaler on train only, so val is scaled with train's statistics rather than its own.
        """

        # Build the splits.
        splits = fake_splits()

        # Shift val so its own mean is far from train's.
        splits["val"][FEATURES] = splits["val"][FEATURES] + 10

        # Fit the model briefly.
        model = TemporalCNNModel(features = FEATURES, target = TARGET, max_epochs = 1).fit(splits = splits)

        # Check the scaler learned train's mean, not val's.
        assert np.allclose(model.scaler.mean_, splits["train"][FEATURES].mean())

    def test_learns_simple_rule(self):
        """
        Reach a high validation F1 on data with an easy rule.
        """

        # Check the rule was learned.
        assert val_f1(TemporalCNNModel(features = FEATURES, target = TARGET, max_epochs = 40, patience = 5)) > 0.8

    def test_early_stopping_restores_best(self):
        """
        Stop patience epochs after the best epoch and restore that epoch's weights.
        """

        # Build the splits.
        splits = fake_splits()

        # Fit the model.
        model = TemporalCNNModel(features = FEATURES, target = TARGET, max_epochs = 40, patience = 5).fit(splits = splits)

        # Check it stopped patience epochs after the best, or at the epoch limit.
        assert len(model.history) in (model.best_epoch + 5, 40)

        # Build the validation windows.
        val = model.windows(splits = splits)["val"]

        # Recalculate the validation loss of the restored network.
        with torch.no_grad():

            # Score the validation windows.
            val_loss = nn.CrossEntropyLoss()(model.network(torch.tensor(val["X"])), torch.tensor(val["y"], dtype = torch.long)).item()

        # Check the restored network has the best epoch's validation loss.
        assert val_loss == pytest.approx(model.history[model.best_epoch - 1]["val_loss"], rel = 1e-5)
