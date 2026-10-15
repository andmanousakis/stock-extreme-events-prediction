from pathlib import Path
import pandas as pd
from utils.config_loader import ConfigLoader
from utils.error_handler import log_step
from utils.logger import Logger
from utils.schema_loader import SchemaLoader

class DataPreprocessing:
    """
    Clean the raw stock price data and label it: validate it, keep one row per trading day, fill any missing values,
    calculate daily returns, label extreme events and shift the target to the next day.
    """

    def __init__(self):
        """
        Create the logger and load the data preprocessing configs and raw data schema.
        """

        # Create a logger named after the class.
        self.logger = Logger().get_logger(self.__class__.__name__)

        # Load configs.
        self.config = ConfigLoader().load("config-preprocessing")

        # Get the input configs.
        self.input = self.config["input"]

        # Get the missing values configs.
        self.missing_values = self.config["missing_values"]

        # Get the daily returns configs.
        self.daily_returns_config = self.config["daily_returns"]

        # Get the extreme events configs.
        self.extreme_events_config = self.config["extreme_events"]

        # Get the target configs.
        self.target = self.config["target"]

        # Get the output configs.
        self.output = self.config["output"]

        # Create the schema loader.
        self.schema_loader = SchemaLoader()

        # Load the raw data schema.
        self.schema = self.schema_loader.load(self.input["schema"])

    @log_step("Load raw data")
    def load_raw_data(self, path: str = None) -> pd.DataFrame:
        """
        Load the raw price data stored by data retrieval.

        Args:
            path (str): Raw data file path.

        Returns:
            pd.DataFrame: Raw price data.
        """

        # Load raw data.
        df = pd.read_excel(path)

        # Return data.
        return df

    @log_step("Schema validation")
    def apply_schema(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Check the raw data against the raw data schema and cast its columns to the schema dtypes.

        Args:
            df (pd.DataFrame): Raw price data.

        Returns:
            pd.DataFrame: Data matching the schema.
        """

        # Apply the schema.
        df = self.schema_loader.apply(df = df, schema = self.schema)

        # Return data.
        return df

    @log_step("Sort and deduplicate")
    def sort_and_deduplicate(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Sort rows by date and keep one row per date.

        Args:
            df (pd.DataFrame): Price data with a 'Date' column.

        Returns:
            pd.DataFrame: Data sorted by date with unique dates.
        """

        # Sort rows by date.
        df = df.sort_values("Date")

        # Count duplicate dates.
        duplicates = int(df["Date"].duplicated(keep = "last").sum())

        # Keep the last row for each date.
        df = df.drop_duplicates(subset = "Date", keep = "last").reset_index(drop = True)

        # Log removed duplicates.
        self.logger.info(f"Removed {duplicates} duplicate dates.")

        # Return data.
        return df

    @log_step("Missing values handling")
    def handle_missing_values(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Forward-fill missing values on trading days. Weekends and holidays are not added, since no price exists for them.

        Args:
            df (pd.DataFrame): Price data with one row per trading day.

        Returns:
            pd.DataFrame: Data without missing values.
        """

        # Get the columns to fill.
        columns = self.missing_values["columns"]

        # Count missing values before filling.
        missing = int(df[columns].isna().sum().sum())

        # Forward-fill gaps with the previous trading day's values.
        df[columns] = df[columns].ffill()

        # Count rows that still have missing values (leading gaps with no previous day).
        unfilled = int(df[columns].isna().any(axis = 1).sum())

        # Drop rows that could not be filled.
        df = df.dropna(subset = columns).reset_index(drop = True)

        # Log filled and dropped values.
        self.logger.info(f"Forward-filled {missing} missing values, dropped {unfilled} unfillable rows.")

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

    @log_step("Store clean data")
    def store_clean_data(self, df: pd.DataFrame = None, path: str = None) -> pd.DataFrame:
        """
        Store the clean price data to Excel.

        Args:
            df (pd.DataFrame): Clean price data.
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
        Run the data preprocessing steps in order.
        """

        # Load raw data.
        df = self.load_raw_data(path = self.input["raw_data"])

        # Check the raw data against the schema.
        df = self.apply_schema(df = df)

        # Sort by date and remove duplicate dates.
        df = self.sort_and_deduplicate(df = df)

        # Fill missing values on trading days.
        df = self.handle_missing_values(df = df)

        # Calculate daily returns.
        df = self.daily_returns(df = df)

        # Label extreme events.
        df = self.extreme_events(df = df)

        # Shift the target to the next day.
        df = self.shift_target(df = df)

        # Store clean data.
        self.store_clean_data(df = df, path = self.output["clean_data"])

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

    # Run data preprocessing.
    DataPreprocessing.run()
