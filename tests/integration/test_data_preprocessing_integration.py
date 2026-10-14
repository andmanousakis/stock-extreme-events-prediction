import importlib
import pandas as pd
import pytest

data_retrieval = importlib.import_module("src.1_data_retrieval")
data_preprocessing = importlib.import_module("src.2_data_preprocessing")

@pytest.mark.integration
class TestDataPreprocessingIntegration:
    """
    Integration tests for DataPreprocessing on real data produced by DataRetrieval.
    """

    def test_pipeline_cleans_retrieved_data(self, workdir):
        """
        Retrieve real data, preprocess it, and check the clean file is valid, sorted, unique and complete.

        Args:
            workdir (Path): Temporary folder both pipelines write to.
        """

        # Create the data retrieval object.
        retrieval = data_retrieval.DataRetrieval()

        # Create the data preprocessing object.
        preprocessing = data_preprocessing.DataPreprocessing()

        # Check preprocessing reads the file retrieval writes.
        assert preprocessing.input["raw_data"] == retrieval.output["raw_data"]

        # Retrieve real data.
        retrieval.pipeline()

        # Run the full preprocessing pipeline on the retrieved file.
        preprocessing.pipeline()

        # Build the path of the clean file.
        path = workdir / preprocessing.output["clean_data"]

        # Check the file was written.
        assert path.exists()

        # Read the raw file.
        raw = pd.read_excel(workdir / preprocessing.input["raw_data"])

        # Read the clean file.
        clean = pd.read_excel(path)

        # Check the clean data still passes the schema.
        preprocessing.schema_loader.apply(df = clean, schema = preprocessing.schema)

        # Check one row per trading day and none added or lost.
        assert len(clean) == raw["Date"].nunique()

        # Check the dates are sorted.
        assert clean["Date"].is_monotonic_increasing

        # Check the dates are unique.
        assert clean["Date"].is_unique

        # Check there are no missing values.
        assert not clean.isna().any().any()
