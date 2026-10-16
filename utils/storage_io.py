import os
from datetime import datetime, timezone
import fsspec
import pandas as pd
import s3fs

class StorageIO:
    """
    Read and write pandas DataFrames as Parquet tables, one folder per environment and table, one file per run.
    Tables live on the S3-compatible bucket named by STORAGE_BUCKET (DagsHub storage) when it is set,
    and under the local data folder otherwise. Earlier runs' files are kept, so history stays readable.
    """

    def __init__(self, bucket: str = None, env: str = None, local_dir: str = "data"):
        """
        Set where tables live, the run that writes them, and the S3 connection from the environment.

        Args:
            bucket (str): S3 bucket name. Defaults to the STORAGE_BUCKET environment variable; local storage if neither is set.
            env (str): Environment folder. Defaults to the APP_ENV environment variable, or dev if it is not set.
            local_dir (str): Local folder used when no bucket is set.
        """

        # Use the given bucket, then STORAGE_BUCKET.
        self.bucket = bucket or os.environ.get("STORAGE_BUCKET")

        # Use the given environment, then APP_ENV, then dev.
        self.env = env or os.environ.get("APP_ENV", "dev")

        # Get the write time in UTC, used to order files.
        self.stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

        # Use the CI run ID if given, otherwise a local one from the write time.
        self.run_id = os.environ.get("PIPELINE_RUN_ID") or f"local-{self.stamp}"

        # Connect to the bucket if one is set, otherwise use the local file system.
        self.fs = s3fs.S3FileSystem(key = os.environ.get("AWS_ACCESS_KEY_ID"), secret = os.environ.get("AWS_SECRET_ACCESS_KEY"), client_kwargs = {"endpoint_url": os.environ.get("AWS_ENDPOINT_URL"), "region_name": os.environ.get("AWS_REGION", "us-east-1")}) if self.bucket else fsspec.filesystem("file", auto_mkdir = True)

        # Set the root folder for this environment's tables.
        self.root = f"{self.bucket}/{self.env}" if self.bucket else os.path.join(local_dir, self.env)

    def path(self, table: str = None) -> str:
        """
        Build a table's folder.

        Args:
            table (str): Table name, e.g. 'raw/apple_stock'.

        Returns:
            str: Table folder.
        """

        # Return the table folder under the environment root.
        return f"{self.root}/{table}"

    def files(self, table: str = None) -> list:
        """
        List a table's Parquet files, oldest first.

        Args:
            table (str): Table name.

        Returns:
            list: File paths sorted by write time.
        """

        # List the table folder without cached results, keeping Parquet files.
        return sorted(f for f in self.fs.ls(self.path(table = table), detail = False, refresh = True) if f.endswith(".parquet")) if self.fs.exists(self.path(table = table)) else []

    def write(self, df: pd.DataFrame = None, table: str = None) -> str:
        """
        Write a DataFrame as this run's file of the table, keeping earlier runs' files.

        Args:
            df (pd.DataFrame): Data to write.
            table (str): Table name, e.g. 'raw/apple_stock'.

        Returns:
            str: The written file path.
        """

        # Name the file by write time and run so files sort chronologically.
        path = f"{self.path(table = table)}/{self.stamp}_{self.run_id}.parquet"

        # Open the file for writing.
        with self.fs.open(path, "wb") as file:

            # Write the data as Parquet.
            df.to_parquet(file, index = False)

        # Return the file path.
        return path

    def read(self, table: str = None, run_id: str = None, all_runs: bool = False) -> pd.DataFrame:
        """
        Read a table: the latest run by default, a given run, or every run combined.

        Args:
            table (str): Table name.
            run_id (str): Run to read; the latest if not given.
            all_runs (bool): Combine every run's file instead of reading one.

        Returns:
            pd.DataFrame: The table's data.

        Raises:
            FileNotFoundError: If the table has no matching files.
        """

        # List the table's files.
        files = self.files(table = table)

        # Keep the given run's files, every file, or the latest one.
        files = [f for f in files if f.endswith(f"_{run_id}.parquet")] if run_id else files if all_runs else files[-1:]

        # Fail clearly if nothing matches.
        if not files:

            # Raise error.
            raise FileNotFoundError(f"No data for table '{table}' at {self.path(table = table)}" + (f" for run {run_id}" if run_id else ""))

        # Read and combine the files.
        return pd.concat([self.read_file(path = f) for f in files], ignore_index = True)

    def read_file(self, path: str = None) -> pd.DataFrame:
        """
        Read one Parquet file.

        Args:
            path (str): File path.

        Returns:
            pd.DataFrame: The file's data.
        """

        # Open the file for reading.
        with self.fs.open(path, "rb") as file:

            # Read the Parquet data.
            return pd.read_parquet(file)
