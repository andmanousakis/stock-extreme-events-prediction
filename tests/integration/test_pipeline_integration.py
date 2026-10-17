import importlib
import joblib
import pandas as pd
import pytest
from utils.storage_io import StorageIO

data_retrieval = importlib.import_module("src.pipeline.1_data_retrieval")
data_preprocessing = importlib.import_module("src.pipeline.2_data_preprocessing")
feature_engineering = importlib.import_module("src.pipeline.3_feature_engineering")
data_split = importlib.import_module("src.pipeline.4_data_split")
model_training_evaluation = importlib.import_module("src.pipeline.5_model_training_evaluation")
model_selection = importlib.import_module("src.pipeline.6_model_selection")
model_monitoring = importlib.import_module("src.pipeline.7_model_monitoring")

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
        training = model_training_evaluation.ModelTrainingEvaluation()
        selection = model_selection.ModelSelection()
        monitoring = model_monitoring.ModelMonitoring()

        # Check preprocessing reads the file retrieval writes.
        assert preprocessing.input["raw_data"] == retrieval.output["raw_data"]

        # Check feature engineering reads the file preprocessing writes.
        assert engineering.input["clean_data"] == preprocessing.output["clean_data"]

        # Check the split reads the file feature engineering writes.
        assert splitter.input["features"] == engineering.output["features"]

        # Check training reads the files the split writes.
        assert all(training.input[name] == splitter.output[name] for name in ("train", "val", "test"))

        # Check selection reads the metrics table and models training writes.
        assert selection.input["metrics_table"] == training.output["tables"]["metrics"] and selection.input["models_dir"] == training.output["models_dir"]

        # Check monitoring reads the splits and metrics table training writes, and the model selection registers.
        assert all(monitoring.input[name] == training.input[name] for name in ("train", "val", "test")) and monitoring.input["metrics_table"] == training.output["tables"]["metrics"] and monitoring.tracker.registered_model == selection.tracker.registered_model

        # Check every step uses the same target column.
        assert preprocessing.target["column"] == engineering.target["column"] == splitter.target["column"] == training.target["column"]

        # Run every step in order.
        for step in (retrieval, preprocessing, engineering, splitter, training, selection, monitoring):

            # Run the step.
            step.pipeline()

        # Read the raw data.
        raw = StorageIO().read(table = preprocessing.input["raw_data"])

        # Read the feature dataset.
        features = StorageIO().read(table = engineering.output["features"])

        # Recalculate each day's extreme event from the raw data.
        events = (raw[preprocessing.daily_returns_config["price_column"]].pct_change().mul(100).abs() > preprocessing.extreme_events_config["threshold"]).astype(int)

        # Pair each date with the next day's extreme event.
        expected = pd.Series(events.shift(-1).to_numpy(), index = raw["Date"])

        # Check each target is the next day's extreme event.
        assert features[engineering.target["column"]].tolist() == expected.loc[features["Date"]].astype(int).tolist()

        # Read the splits.
        splits = {name: StorageIO().read(table = splitter.output[name]) for name in ("train", "val", "test")}

        # Check training ends before validation starts and validation ends before test starts.
        assert splits["train"]["Date"].max() < splits["val"]["Date"].min() and splits["val"]["Date"].max() < splits["test"]["Date"].min()

        # Check the splits put back together equal the feature dataset.
        assert pd.concat(splits.values(), ignore_index = True).equals(features)

        # Load the best model.
        best = joblib.load(workdir / selection.output["best_model"])

        # Check the best model predicts once per test day.
        assert len(best.predict(splits = splits)["test"]["pred"]) == len(splits["test"])

        # Read the metrics table.
        metrics = StorageIO().read(table = training.output["tables"]["metrics"], all_runs = True)

        # Check every configured model and split was scored and tagged with the run.
        assert set(training.models_config) <= set(metrics["model"]) and set(metrics["split"]) == {"train", "val", "test"} and (metrics["run_id"] == training.run_id).all()
