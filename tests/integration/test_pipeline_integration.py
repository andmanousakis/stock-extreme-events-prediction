import importlib
import joblib
import pandas as pd
import pytest

data_retrieval = importlib.import_module("src.1_data_retrieval")
data_preprocessing = importlib.import_module("src.2_data_preprocessing")
feature_engineering = importlib.import_module("src.3_feature_engineering")
data_split = importlib.import_module("src.4_data_split")
model_selection = importlib.import_module("src.5_model_selection")

@pytest.mark.integration
class TestPipelineIntegration:
    """
    End-to-end test of the full pipeline on real Yahoo Finance data.
    """

    def test_full_pipeline(self, workdir):
        """
        Run every step on real data, check each step reads the previous step's output, the target is the next day's event,
        the split is time-ordered without overlap or loss, and the best saved model predicts on the real test days.

        Args:
            workdir (Path): Temporary folder all steps write to.
        """

        # Create every step.
        retrieval = data_retrieval.DataRetrieval()
        preprocessing = data_preprocessing.DataPreprocessing()
        engineering = feature_engineering.FeatureEngineering()
        splitter = data_split.DataSplit()
        selection = model_selection.ModelSelection()

        # Check preprocessing reads the file retrieval writes.
        assert preprocessing.input["raw_data"] == retrieval.output["raw_data"]

        # Check feature engineering reads the file preprocessing writes.
        assert engineering.input["clean_data"] == preprocessing.output["clean_data"]

        # Check the split reads the file feature engineering writes.
        assert splitter.input["features"] == engineering.output["features"]

        # Check model selection reads the files the split writes.
        assert all(selection.input[name] == splitter.output[name] for name in ("train", "val", "test"))

        # Check every step uses the same target column.
        assert preprocessing.target["column"] == engineering.target["column"] == splitter.target["column"] == selection.target["column"]

        # Run every step in order.
        for step in (retrieval, preprocessing, engineering, splitter, selection):

            # Run the step.
            step.pipeline()

        # Read the raw data.
        raw = pd.read_excel(workdir / preprocessing.input["raw_data"])

        # Read the feature dataset.
        features = pd.read_excel(workdir / engineering.output["features"])

        # Recalculate each day's extreme event from the raw data.
        events = (raw[preprocessing.daily_returns_config["price_column"]].pct_change().mul(100).abs() > preprocessing.extreme_events_config["threshold"]).astype(int)

        # Pair each date with the next day's extreme event.
        expected = pd.Series(events.shift(-1).to_numpy(), index = raw["Date"])

        # Check each target is the next day's extreme event.
        assert features[engineering.target["column"]].tolist() == expected.loc[features["Date"]].astype(int).tolist()

        # Read the splits.
        splits = {name: pd.read_excel(workdir / splitter.output[name]) for name in ("train", "val", "test")}

        # Check training ends before validation starts and validation ends before test starts.
        assert splits["train"]["Date"].max() < splits["val"]["Date"].min() and splits["val"]["Date"].max() < splits["test"]["Date"].min()

        # Check the splits put back together equal the feature dataset.
        assert pd.concat(splits.values(), ignore_index = True).equals(features)

        # Load the best model.
        best = joblib.load(workdir / selection.output["best_model"])

        # Check the best model predicts once per test day.
        assert len(best.predict(splits = splits)["test"]["pred"]) == len(splits["test"])

        # Read the evaluation report.
        report = pd.read_csv(workdir / selection.output["report"])

        # Check every configured model and split was scored.
        assert set(selection.models_config) <= set(report["model"]) and set(report["split"]) == {"train", "val", "test"}
