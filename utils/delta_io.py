import os
import pandas as pd
from deltalake import DeltaTable, write_deltalake

class DeltaIO:
    """
    Read and write pandas DataFrames as Delta tables, one folder per environment. Tables live on the S3 bucket
    named by DELTA_BUCKET (S3-compatible storage, e.g. Backblaze B2) when it is set, and under the local data folder otherwise.
    Every write adds a new table version, so earlier versions stay readable.
    """

    def __init__(self, bucket: str = None, env: str = None, local_dir: str = "data"):
        """
        Set where tables live and the S3 settings from the environment.

        Args:
            bucket (str): S3 bucket name. Defaults to the DELTA_BUCKET environment variable; local storage if neither is set.
            env (str): Environment folder. Defaults to the APP_ENV environment variable, or dev if it is not set.
            local_dir (str): Local folder used when no bucket is set.
        """

        # Use the given bucket, then DELTA_BUCKET.
        self.bucket = bucket or os.environ.get("DELTA_BUCKET")

        # Use the given environment, then APP_ENV, then dev.
        self.env = env or os.environ.get("APP_ENV", "dev")

        # Store the local fallback folder.
        self.local_dir = local_dir

        # Pass the S3 endpoint, keys and region to Delta, without a locking service (one writer per table).
        self.storage_options = {
            "AWS_ENDPOINT_URL": os.environ.get("AWS_ENDPOINT_URL", ""),
            "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID", ""),
            "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
            "AWS_REGION": os.environ.get("AWS_REGION", "us-east-1"),
            "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
        } if self.bucket else {}

    def uri(self, table: str = None) -> str:
        """
        Build a table's location: s3://<bucket>/<env>/<table> on the bucket, or <local_dir>/<env>/<table> locally.

        Args:
            table (str): Table name, e.g. 'raw/apple_stock'.

        Returns:
            str: Table location.
        """

        # Return the bucket location if a bucket is set.
        if self.bucket:

            # Return the S3 location.
            return f"s3://{self.bucket}/{self.env}/{table}"

        # Return the local location.
        return os.path.join(self.local_dir, self.env, table)

    def write(self, df: pd.DataFrame = None, table: str = None, mode: str = "overwrite") -> int:
        """
        Write a DataFrame as a new version of the table: 'overwrite' replaces its contents and schema, 'append' adds rows.

        Args:
            df (pd.DataFrame): Data to write.
            table (str): Table name, e.g. 'raw/apple_stock'.
            mode (str): 'overwrite' or 'append'.

        Returns:
            int: The new table version.
        """

        # Write the data as a new version, letting the schema change on overwrite and grow on append.
        write_deltalake(self.uri(table = table), df, mode = mode, schema_mode = "overwrite" if mode == "overwrite" else "merge", storage_options = self.storage_options)

        # Return the new version number.
        return DeltaTable(self.uri(table = table), storage_options = self.storage_options).version()

    def read(self, table: str = None, version: int = None) -> pd.DataFrame:
        """
        Read a table, the latest version or an earlier one.

        Args:
            table (str): Table name, e.g. 'raw/apple_stock'.
            version (int): Table version to read; the latest if not given.

        Returns:
            pd.DataFrame: The table's data.
        """

        # Read the requested version.
        return DeltaTable(self.uri(table = table), version = version, storage_options = self.storage_options).to_pandas()
