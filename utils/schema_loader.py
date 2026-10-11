from pathlib import Path
import pandas as pd
import yaml

class SchemaLoader:
    """
    Load YAML table schemas from the project's schemas directory and apply them to DataFrames.
    """

    def __init__(self, schema_dir: str = None):
        """
        Set the directory that schema files are read from.

        Args:
            schema_dir (str): Path to the schemas directory. Defaults to <project_root>/schemas.
        """

        # Default to the schemas directory at the project root.
        self.schema_dir = Path(schema_dir) if schema_dir else Path(__file__).resolve().parents[1] / "schemas"

    def load(self, name: str = None) -> dict:
        """
        Read a schema file and return its contents.

        Args:
            name (str): Schema file name without the .yaml extension.

        Returns:
            dict: Parsed schema with a 'columns' mapping of column name to dtype.
        """

        # Build the schema file path.
        path = self.schema_dir / f"{name}.yaml"

        # Open the schema file.
        with open(path, "r") as file:

            # Parse file contents.
            schema = yaml.safe_load(file)

        # Return schema.
        return schema

    def apply(self, df: pd.DataFrame = None, schema: dict = None) -> pd.DataFrame:
        """
        Keep only the schema's columns, in schema order, and cast them to the schema's dtypes.

        Args:
            df (pd.DataFrame): Data to check.
            schema (dict): Schema with a 'columns' mapping of column name to dtype.

        Returns:
            pd.DataFrame: Data matching the schema.

        Raises:
            ValueError: If a schema column is missing from the data.
            TypeError: If a column cannot be cast to its schema dtype.
        """

        # Get the expected columns and dtypes.
        columns = schema["columns"]

        # Find schema columns missing from the data.
        missing = [column for column in columns if column not in df.columns]

        # Fail if any schema column is missing.
        if missing:

            # Raise error.
            raise ValueError(f"Missing columns: {missing}")

        # Select the schema columns in schema order.
        df = df[list(columns)]

        try:

            # Cast columns to their schema dtypes.
            df = df.astype(columns)

        except (ValueError, TypeError) as e:

            # Raise error naming the schema problem.
            raise TypeError(f"Data does not match schema dtypes: {e}") from e

        # Return data.
        return df
