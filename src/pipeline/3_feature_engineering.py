import pandas as pd
from utils.config_loader import ConfigLoader
from utils.storage_io import StorageIO
from utils.error_handler import log_step
from utils.logger import Logger

class FeatureEngineering:
    """
    Add model features to the clean, labelled data: volatility, intraday range, overnight gap and relative volume.
    """

    def __init__(self):
        """
        Create the logger and load the feature engineering configs.
        """

        # Create a logger named after the class.
        self.logger = Logger().get_logger(self.__class__.__name__)

        # Load configs.
        self.config = ConfigLoader().load("config-features")

        # Create the table storage.
        self.storage = StorageIO()

        # Get the input configs.
        self.input = self.config["input"]

        # Get the extra features configs.
        self.extra_features_config = self.config["extra_features"]

        # Get the target configs.
        self.target = self.config["target"]

        # Get the feature columns.
        self.features = self.config["features"]

        # Get the output configs.
        self.output = self.config["output"]

    @log_step("Load clean data")
    def load_clean_data(self, path: str = None) -> pd.DataFrame:
        """
        Load the clean price data stored by data preprocessing.

        Args:
            path (str): Clean data table name.

        Returns:
            pd.DataFrame: Clean price data, one row per trading day.
        """

        # Read the clean data table.
        df = self.storage.read(table = path)

        # Return data.
        return df

    @log_step("Extra features calculation")
    def extra_features(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Add volatility, intraday range, overnight gap and relative volume, then drop the first days that lack enough history.
        Every value only uses the current and earlier days.

        Args:
            df (pd.DataFrame): Data with price, volume and 'daily_return' columns.

        Returns:
            pd.DataFrame: Data with the extra feature columns and no missing values.
        """

        # Add the rolling standard deviation of returns for each window.
        for window in self.extra_features_config["volatility_windows"]:

            # Measure how much returns swung over the window.
            df[f"volatility_{window}"] = df["daily_return"].rolling(window).std()

        # Measure the intraday swing as a percentage of the close.
        df["range"] = (df["High"] - df["Low"]) / df["Close"] * 100

        # Measure the jump from yesterday's close to today's open as a percentage.
        df["gap"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1) * 100

        # Compare today's volume with its recent average.
        df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(self.extra_features_config["volume_window"]).mean()

        # Count the first rows without enough history.
        incomplete = int(df[self.features].isna().any(axis = 1).sum())

        # Drop them.
        df = df.dropna(subset = self.features).reset_index(drop = True)

        # Log dropped rows.
        self.logger.info(f"Dropped {incomplete} first days without enough history for the extra features.")

        # Return data.
        return df

    @log_step("Column selection")
    def select_columns(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Keep the date, the feature columns and the target.

        Args:
            df (pd.DataFrame): Data with features and target.

        Returns:
            pd.DataFrame: Date, features and target only.
        """

        # Keep the date, features and target in order.
        df = df[["Date"] + self.features + [self.target["column"]]]

        # Return data.
        return df

    @log_step("Store features")
    def store_features(self, df: pd.DataFrame = None, path: str = None) -> pd.DataFrame:
        """
        Store the feature dataset as a Parquet table.

        Args:
            df (pd.DataFrame): Feature dataset.
            path (str): Output table name.

        Returns:
            pd.DataFrame: The stored data.
        """

        # Write the data as this run's file of the table.
        self.storage.write(df = df, table = path)

        # Return data.
        return df

    def pipeline(self):
        """
        Run the feature engineering steps in order.
        """

        # Load clean data.
        df = self.load_clean_data(path = self.input["clean_data"])

        # Add the extra features.
        df = self.extra_features(df = df)

        # Keep the date, features and target.
        df = self.select_columns(df = df)

        # Store the feature dataset.
        self.store_features(df = df, path = self.output["features"])

    @classmethod
    def run(cls):
        """
        Instantiate the class and run the pipeline.
        """

        # Instantiate class object.
        instance = cls()

        # Run the pipeline.
        instance.pipeline()

if __name__ == "__main__":

    # Run feature engineering.
    FeatureEngineering.run()
