import importlib
import joblib
import pandas as pd
import pytest
import torch
from src.networks.temporal_cnn import TemporalCNNNetwork
from utils.windowing import Windowing

data_retrieval = importlib.import_module("src.1_data_retrieval")
data_preprocessing = importlib.import_module("src.2_data_preprocessing")
feature_engineering = importlib.import_module("src.3_feature_engineering")
data_split = importlib.import_module("src.4_data_split")
random_forest = importlib.import_module("src.5_random_forest")
temporal_cnn = importlib.import_module("src.6_temporal_cnn")

@pytest.mark.integration
class TestPipelineIntegration:
    """
    End-to-end test of the full pipeline on real Yahoo Finance data.
    """

    def test_full_pipeline(self, workdir):
        """
        Run every step on real data, check each step reads the previous step's output, the target is the next day's event,
        the split is time-ordered without overlap or loss, and both saved models predict on the real test days.

        Args:
            workdir (Path): Temporary folder all steps write to.
        """

        # Create every step.
        retrieval = data_retrieval.DataRetrieval()
        preprocessing = data_preprocessing.DataPreprocessing()
        engineering = feature_engineering.FeatureEngineering()
        splitter = data_split.DataSplit()
        forest = random_forest.RandomForest()
        cnn = temporal_cnn.TemporalCNN()

        # Check preprocessing reads the file retrieval writes.
        assert preprocessing.input["raw_data"] == retrieval.output["raw_data"]

        # Check feature engineering reads the file preprocessing writes.
        assert engineering.input["clean_data"] == preprocessing.output["clean_data"]

        # Check the split reads the file feature engineering writes.
        assert splitter.input["features"] == engineering.output["features"]

        # Check both models read the files the split writes.
        assert all(model.input[name] == splitter.output[name] for model in (forest, cnn) for name in ("train", "val", "test"))

        # Check every step uses the same target column.
        assert engineering.target["column"] == splitter.target["column"] == forest.target["column"] == cnn.target["column"]

        # Run every step in order.
        for step in (retrieval, preprocessing, engineering, splitter, forest, cnn):

            # Run the step.
            step.pipeline()

        # Read the clean data.
        clean = pd.read_excel(workdir / engineering.input["clean_data"])

        # Read the feature dataset.
        features = pd.read_excel(workdir / engineering.output["features"])

        # Recalculate each day's extreme event from the clean data.
        events = (clean[engineering.daily_returns_config["price_column"]].pct_change().mul(100).abs() > engineering.extreme_events_config["threshold"]).astype(int)

        # Pair each date with the next day's extreme event.
        expected = pd.Series(events.shift(-1).to_numpy(), index = clean["Date"])

        # Check each target is the next day's extreme event.
        assert features[engineering.target["column"]].tolist() == expected.loc[features["Date"]].astype(int).tolist()

        # Read the splits.
        splits = {name: pd.read_excel(workdir / splitter.output[name]) for name in ("train", "val", "test")}

        # Check training ends before validation starts and validation ends before test starts.
        assert splits["train"]["Date"].max() < splits["val"]["Date"].min() and splits["val"]["Date"].max() < splits["test"]["Date"].min()

        # Check the splits put back together equal the feature dataset.
        assert pd.concat(splits.values(), ignore_index = True).equals(features)

        # Load the random forest.
        rf = joblib.load(workdir / forest.output["model"])

        # Build the random forest's test windows.
        rf_test = Windowing(window = rf["window"]).build(splits = splits, features = rf["features"], target = forest.target["column"])["test"]

        # Check the random forest predicts once per test day.
        assert len(rf["model"].predict(rf_test["X"].reshape(len(rf_test["X"]), -1))) == len(splits["test"])

        # Load the temporal CNN.
        tcnn = joblib.load(workdir / cnn.output["model"])

        # Rebuild the network with its saved weights.
        network = TemporalCNNNetwork(num_features = len(tcnn["features"]), window = tcnn["window"], **tcnn["network"])

        # Load the saved weights.
        network.load_state_dict(tcnn["state_dict"])

        # Scale the splits with the saved scaler.
        scaled = {name: df.assign(**dict(zip(tcnn["features"], tcnn["scaler"].transform(df[tcnn["features"]]).T))) for name, df in splits.items()}

        # Build the temporal CNN's test windows shaped [samples, features, days].
        cnn_test = Windowing(window = tcnn["window"]).build(splits = scaled, features = tcnn["features"], target = cnn.target["column"])["test"]["X"].transpose(0, 2, 1)

        # Check the temporal CNN predicts once per test day.
        assert network.predict_proba(torch.tensor(cnn_test, dtype = torch.float32)).shape == (len(splits["test"]), 2)
