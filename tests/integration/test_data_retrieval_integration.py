import importlib
import pandas as pd
import pytest

data_retrieval = importlib.import_module("src.1_data_retrieval")

@pytest.mark.integration
class TestDataRetrievalIntegration:
    """
    Integration tests for DataRetrieval against the real Yahoo Finance API.
    """

    def test_pipeline_stores_valid_raw_data(self, workdir):
        """
        Download real data, store it, and check the stored file matches the schema and requested dates.

        Args:
            workdir (Path): Temporary folder the pipeline writes to.
        """

        # Create the data retrieval object.
        retrieval = data_retrieval.DataRetrieval()

        # Run the full pipeline.
        retrieval.pipeline()

        # Build the path of the stored file.
        path = workdir / retrieval.output["raw_data"]

        # Check the file was written.
        assert path.exists()

        # Read the stored file.
        df = pd.read_excel(path)

        # Check the stored data still passes the schema.
        retrieval.schema_loader.apply(df = df, schema = retrieval.schema)

        # Check there is data.
        assert len(df) > 0

        # Check there are no missing values.
        assert not df.isna().any().any()

        # Check no date is before the requested start.
        assert df["Date"].min() >= pd.Timestamp(retrieval.request["start"])

        # Check no date is on or after the requested end, which yfinance excludes.
        assert df["Date"].max() < pd.Timestamp(retrieval.request["end"])
