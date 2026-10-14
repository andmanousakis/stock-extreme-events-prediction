import importlib
import pandas as pd
import pytest

data_retrieval = importlib.import_module("src.1_data_retrieval")
data_preprocessing = importlib.import_module("src.2_data_preprocessing")
feature_engineering = importlib.import_module("src.3_feature_engineering")
data_split = importlib.import_module("src.4_data_split")

@pytest.mark.integration
class TestDataSplitIntegration:
    """
    Integration tests for DataSplit on real data produced by the retrieval, preprocessing and feature engineering steps.
    """

    def test_pipeline_splits_real_features(self, workdir):
        """
        Run the full chain on real data and check the splits are in order, don't overlap and lose no rows.

        Args:
            workdir (Path): Temporary folder all pipelines write to.
        """

        # Create the feature engineering object.
        engineering = feature_engineering.FeatureEngineering()

        # Create the data split object.
        splitter = data_split.DataSplit()

        # Check the split reads the file feature engineering writes.
        assert splitter.input["features"] == engineering.output["features"]

        # Check the split summarizes the target feature engineering creates.
        assert splitter.target["column"] == engineering.target["column"]

        # Retrieve real data.
        data_retrieval.DataRetrieval().pipeline()

        # Clean the retrieved data.
        data_preprocessing.DataPreprocessing().pipeline()

        # Build features from the clean data.
        engineering.pipeline()

        # Split the features.
        splitter.pipeline()

        # Read the feature dataset.
        features = pd.read_excel(workdir / splitter.input["features"])

        # Read the stored splits.
        train, val, test = (pd.read_excel(workdir / splitter.output[name]) for name in ("train", "val", "test"))

        # Check training ends before validation starts.
        assert train["Date"].max() < val["Date"].min()

        # Check validation ends before test starts.
        assert val["Date"].max() < test["Date"].min()

        # Check the splits put back together equal the feature dataset.
        assert pd.concat([train, val, test], ignore_index = True).equals(features)

        # Check training holds about 70% of the rows.
        assert len(train) == int(len(features) * splitter.ratios["train"])
