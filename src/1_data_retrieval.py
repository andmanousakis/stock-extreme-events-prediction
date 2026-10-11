import time
from pathlib import Path
import pandas as pd
import yfinance as yf
from utils.config_loader import ConfigLoader
from utils.error_handler import log_step
from utils.logger import Logger
from utils.schema_loader import SchemaLoader

class DataRetrieval:
    """
    Fetch raw stock price data from Yahoo Finance and store it.
    """

    def __init__(self):
        """
        Create the logger and load the data retrieval configs and raw data schema.
        """

        # Create a logger named after the class.
        self.logger = Logger().get_logger(self.__class__.__name__)

        # Load configs.
        self.config = ConfigLoader().load("config-retrieval")

        # Get the request configs.
        self.request = self.config["request"]

        # Get the output configs.
        self.output = self.config["output"]

        # Create the schema loader.
        self.schema_loader = SchemaLoader()

        # Load the raw data schema.
        self.schema = self.schema_loader.load(self.request["schema"])

    @log_step("Data retrieval")
    def make_request(self, symbol: str = None, start: str = None, end: str = None) -> pd.DataFrame:
        """
        Download daily price data from Yahoo Finance, retrying with exponential backoff.

        Args:
            symbol (str): Ticker symbol, e.g. "AAPL".
            start (str): Start date in YYYY-MM-DD format.
            end (str): End date in YYYY-MM-DD format.

        Returns:
            pd.DataFrame: Price data with a 'Date' column.

        Raises:
            RuntimeError: If every attempt fails.
        """

        for attempt in range(1, self.request["max_retries"] + 1):

            try:

                # Request data.
                data = yf.download(symbol, start = start, end = end, auto_adjust = False)

                # Fail the attempt if yfinance returned no rows instead of raising.
                if data.empty:

                    # Raise error.
                    raise ValueError(f"No data returned for {symbol}.")

                # Move date from the index to a column.
                data.reset_index(inplace = True)

                # Flatten multi-level column headers by keeping the price names and dropping the ticker level.
                data.columns = data.columns.get_level_values(0)

                # Return data.
                return data

            except Exception as e:

                # Log failed attempt.
                self.logger.warning(f"Attempt {attempt}/{self.request['max_retries']} failed: {e}.")

                # Stop retrying after the last attempt.
                if attempt == self.request["max_retries"]:

                    # Exit the loop.
                    break

                # Double the delay after each attempt, up to the maximum.
                delay = min(self.request["base_delay"] * 2 ** (attempt - 1), self.request["max_delay"])

                # Log the wait.
                self.logger.info(f"Retrying in {delay}s.")

                # Wait before retrying.
                time.sleep(delay)

        # Raise error after all attempts failed.
        raise RuntimeError(f"All {self.request['max_retries']} attempts failed")

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

    @log_step("Store raw data")
    def store_raw_data(self, df: pd.DataFrame = None, path: str = None) -> pd.DataFrame:
        """
        Store the raw price data to Excel.

        Args:
            df (pd.DataFrame): Raw price data.
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
        Run the data retrieval steps in order.
        """

        # Step 1.1: Request data.
        raw_df = self.make_request(symbol = self.request["symbol"], start = self.request["start"], end = self.request["end"])

        # Check the raw data against the schema.
        raw_df = self.apply_schema(df = raw_df)

        # Store raw data.
        self.store_raw_data(df = raw_df, path = self.output["raw_data"])

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

    # Run data retrieval.
    DataRetrieval.run()
