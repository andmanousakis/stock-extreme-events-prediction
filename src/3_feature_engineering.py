from pathlib import Path
import pandas as pd
from utils.config_loader import ConfigLoader
from utils.error_handler import log_step
from utils.logger import Logger

class FeatureEngineering:
    """
    Turn clean trading-day prices into a labelled dataset: daily returns, extreme events and a next-day target.
    """

    def __init__(self):
        """
        Create the logger and load the feature engineering configs.
        """

        # Create a logger named after the class.
        self.logger = Logger().get_logger(self.__class__.__name__)

        # Load configs.
        self.config = ConfigLoader().load("config-features")

        # Get the input configs.
        self.input = self.config["input"]

        # Get the daily returns configs.
        self.daily_returns_config = self.config["daily_returns"]

        # Get the extreme events configs.
        self.extreme_events_config = self.config["extreme_events"]

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
            path (str): Clean data file path.

        Returns:
            pd.DataFrame: Clean price data, one row per trading day.
        """

        # Load clean data.
        df = pd.read_excel(path)

        # Return data.
        return df

    @log_step("Daily returns calculation")
    def daily_returns(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Calculate the daily percentage return from one trading day to the next.

        Args:
            df (pd.DataFrame): Clean price data sorted by date.

        Returns:
            pd.DataFrame: Data with a 'daily_return' column, first row dropped.
        """

        # Calculate percentage change of the configured price column.
        df["daily_return"] = df[self.daily_returns_config["price_column"]].pct_change() * 100

        # Drop the first row, which has no previous day.
        df = df.iloc[1:].reset_index(drop = True)

        # Return data.
        return df

    @log_step("Extreme events labelling")
    def extreme_events(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Label days whose absolute daily return exceeds the threshold as extreme events.

        Args:
            df (pd.DataFrame): Data with a 'daily_return' column.

        Returns:
            pd.DataFrame: Data with a binary 'extreme_event' column.
        """

        # Label days beyond the threshold in either direction.
        df["extreme_event"] = (df["daily_return"].abs() > self.extreme_events_config["threshold"]).astype(int)

        # Return data.
        return df

    @log_step("Target shift")
    def shift_target(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Create the target as the extreme event of a later day, so each row predicts the future from today's data.

        Args:
            df (pd.DataFrame): Data with an 'extreme_event' column.

        Returns:
            pd.DataFrame: Data with the target column, last rows without a future label dropped.
        """

        # Move each day's label back by the configured number of days.
        df[self.target["column"]] = df["extreme_event"].shift(-self.target["shift"])

        # Drop the last rows, which have no future day to label them.
        df = df.dropna(subset = [self.target["column"]]).reset_index(drop = True)

        # Cast the target back to integers.
        df[self.target["column"]] = df[self.target["column"]].astype(int)

        # Count extreme days in the target.
        extreme = int(df[self.target["column"]].sum())

        # Log the class balance.
        self.logger.info(f"Target: {extreme} of {len(df)} days are extreme ({extreme / len(df):.1%}).")

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
        Store the feature dataset to Excel.

        Args:
            df (pd.DataFrame): Feature dataset.
            path (str): Output file path.

        Returns:
            pd.DataFrame: The stored data.
        """

        # Create the output directory if it does not exist.
        Path(path).parent.mkdir(parents = True, exist_ok = True)

        # Store data.
        df.to_excel(path, index = False)

        # Return data.
        return df

    def pipeline(self):
        """
        Run the feature engineering steps in order.
        """

        # Load clean data.
        df = self.load_clean_data(path = self.input["clean_data"])

        # Calculate daily returns.
        df = self.daily_returns(df = df)

        # Label extreme events.
        df = self.extreme_events(df = df)

        # Shift the target to the next day.
        df = self.shift_target(df = df)

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
