import importlib
import pandas as pd
import pytest

data_retrieval = importlib.import_module("src.1_data_retrieval")
data_preprocessing = importlib.import_module("src.2_data_preprocessing")
feature_engineering = importlib.import_module("src.3_feature_engineering")

@pytest.mark.integration
class TestFeatureEngineeringIntegration:
    """
    Integration tests for FeatureEngineering on real data produced by DataRetrieval and DataPreprocessing.
    """

    def test_pipeline_builds_features_from_retrieved_data(self, workdir):
        """
        Retrieve and clean real data, build features, and check the targets are the next day's extreme events.

        Args:
            workdir (Path): Temporary folder all pipelines write to.
        """

        # Create the data preprocessing object.
        preprocessing = data_preprocessing.DataPreprocessing()

        # Create the feature engineering object.
        engineering = feature_engineering.FeatureEngineering()

        # Check feature engineering reads the file preprocessing writes.
        assert engineering.input["clean_data"] == preprocessing.output["clean_data"]

        # Retrieve real data.
        data_retrieval.DataRetrieval().pipeline()

        # Clean the retrieved data.
        preprocessing.pipeline()

        # Build features from the clean data.
        engineering.pipeline()

        # Read the clean data.
        clean = pd.read_excel(workdir / engineering.input["clean_data"])

        # Read the feature dataset.
        features = pd.read_excel(workdir / engineering.output["features"])

        # Recalculate daily returns from the clean data.
        returns = clean[engineering.daily_returns_config["price_column"]].pct_change() * 100

        # Recalculate each day's extreme event.
        events = (returns.abs() > engineering.extreme_events_config["threshold"]).astype(int)

        # Build the expected next-day targets, without the first and last days.
        expected = events.shift(-engineering.target["shift"]).iloc[1:-engineering.target["shift"]].astype(int).reset_index(drop = True)

        # Check the columns.
        assert list(features.columns) == ["Date"] + engineering.features + [engineering.target["column"]]

        # Check the first and last days were dropped.
        assert len(features) == len(clean) - 1 - engineering.target["shift"]

        # Check each target is the next day's extreme event.
        assert features[engineering.target["column"]].tolist() == expected.tolist()

        # Check there are no missing values.
        assert not features.isna().any().any()
