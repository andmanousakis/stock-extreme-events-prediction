import pytest
from utils.schema_loader import SchemaLoader

class TestSchemaLoader:
    """
    Unit tests for SchemaLoader.
    """

    def test_load_raw_stock_schema(self):
        """
        Load the project's raw_stock schema and check its columns and dtypes.
        """

        # Load the raw stock schema.
        schema = SchemaLoader().load("raw_stock")

        # Check the column names and order.
        assert list(schema["columns"]) == ["Date", "Adj Close", "Close", "High", "Low", "Open", "Volume"]

        # Check the date column dtype.
        assert schema["columns"]["Date"] == "datetime64[ns]"

        # Check the volume column dtype.
        assert schema["columns"]["Volume"] == "int64"

    def test_load_from_custom_dir(self, tmp_path):
        """
        Load a schema from a directory passed to the constructor.

        Args:
            tmp_path (pathlib.Path): Temporary directory provided by pytest.
        """

        # Write a small schema file.
        (tmp_path / "prices.yaml").write_text("columns:\n  Close: float64\n")

        # Load the schema from the temporary directory.
        schema = SchemaLoader(schema_dir = str(tmp_path)).load("prices")

        # Check the loaded schema.
        assert schema == {"columns": {"Close": "float64"}}

    def test_load_missing_schema_raises(self, tmp_path):
        """
        Raise FileNotFoundError when the schema file does not exist.

        Args:
            tmp_path (pathlib.Path): Temporary directory provided by pytest.
        """

        # Check that loading a missing schema fails.
        with pytest.raises(FileNotFoundError):

            # Load a schema that does not exist.
            SchemaLoader(schema_dir = str(tmp_path)).load("missing")
