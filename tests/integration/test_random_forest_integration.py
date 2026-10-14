import importlib
import joblib
import pandas as pd
import pytest
from utils.windowing import Windowing

data_retrieval = importlib.import_module("src.1_data_retrieval")
data_preprocessing = importlib.import_module("src.2_data_preprocessing")
feature_engineering = importlib.import_module("src.3_feature_engineering")
data_split = importlib.import_module("src.4_data_split")
random_forest = importlib.import_module("src.5_random_forest")

@pytest.mark.integration
class TestRandomForestIntegration:
    """
    Integration tests for RandomForest on real data produced by the full data pipeline.
    """

    def test_pipeline_trains_on_real_splits(self, workdir):
        """
        Run the full chain on real data, train the random forest, and check the saved model predicts on real test windows.

        Args:
            workdir (Path): Temporary folder all pipelines write to.
        """

        # Create the data split object.
        splitter = data_split.DataSplit()

        # Create the random forest object.
        forest = random_forest.RandomForest()

        # Check the random forest reads the files the split writes.
        assert all(forest.input[name] == splitter.output[name] for name in ("train", "val", "test"))

        # Check the random forest predicts the target the split stores.
        assert forest.target["column"] == splitter.target["column"]

        # Retrieve real data.
        data_retrieval.DataRetrieval().pipeline()

        # Clean the retrieved data.
        data_preprocessing.DataPreprocessing().pipeline()

        # Build features from the clean data.
        feature_engineering.FeatureEngineering().pipeline()

        # Split the features.
        splitter.pipeline()

        # Train the random forest.
        forest.pipeline()

        # Load the saved model bundle.
        bundle = joblib.load(workdir / forest.output["model"])

        # Load the real splits.
        splits = {name: pd.read_excel(workdir / splitter.output[name]) for name in ("train", "val", "test")}

        # Build the test windows the same way training did.
        test = Windowing(window = bundle["window"]).build(splits = splits, features = bundle["features"], target = forest.target["column"])["test"]

        # Predict on the flattened test windows.
        predictions = bundle["model"].predict(test["X"].reshape(len(test["X"]), -1))

        # Check there is one prediction per test day.
        assert len(predictions) == len(splits["test"])

        # Check predictions are binary labels.
        assert set(predictions.tolist()) <= {0, 1}

        # Check the validation F1 is a valid score.
        assert 0.0 <= bundle["val_f1"] <= 1.0
