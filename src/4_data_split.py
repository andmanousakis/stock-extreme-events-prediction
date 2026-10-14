from pathlib import Path
import pandas as pd
from utils.config_loader import ConfigLoader
from utils.error_handler import log_step
from utils.logger import Logger

class DataSplit:
    """
    Split the feature dataset in time order into training, validation and test sets.
    """

    def __init__(self):
        """
        Create the logger and load the data split configs.
        """

        # Create a logger named after the class.
        self.logger = Logger().get_logger(self.__class__.__name__)

        # Load configs.
        self.config = ConfigLoader().load("config-split")

        # Get the input configs.
        self.input = self.config["input"]

        # Get the target configs.
        self.target = self.config["target"]

        # Get the split ratios.
        self.ratios = self.config["ratios"]

        # Get the output configs.
        self.output = self.config["output"]

    @log_step("Load features")
    def load_features(self, path: str = None) -> pd.DataFrame:
        """
        Load the feature dataset stored by feature engineering.

        Args:
            path (str): Feature dataset file path.

        Returns:
            pd.DataFrame: Feature dataset sorted by date.
        """

        # Load the feature dataset.
        df = pd.read_excel(path)

        # Make sure rows are in date order before splitting.
        df = df.sort_values("Date").reset_index(drop = True)

        # Return data.
        return df

    @log_step("Time-ordered split")
    def split_data(self, df: pd.DataFrame = None) -> dict:
        """
        Split rows in date order: the oldest go to training, the next to validation and the newest to test.

        Args:
            df (pd.DataFrame): Feature dataset sorted by date.

        Returns:
            dict: 'train', 'val' and 'test' DataFrames.
        """

        # Get the number of rows.
        n = len(df)

        # Calculate where the training set ends.
        train_end = int(n * self.ratios["train"])

        # Calculate where the validation set ends.
        val_end = train_end + int(n * self.ratios["val"])

        # Return the three consecutive slices.
        return {
            "train": df.iloc[:train_end].reset_index(drop = True),
            "val": df.iloc[train_end:val_end].reset_index(drop = True),
            "test": df.iloc[val_end:].reset_index(drop = True),
        }

    @log_step("Split summary")
    def summarize_splits(self, splits: dict = None) -> dict:
        """
        Log each split's row count, date range and share of extreme days.

        Args:
            splits (dict): 'train', 'val' and 'test' DataFrames.

        Returns:
            dict: The same splits, unchanged.
        """

        # Log each split.
        for name, df in splits.items():

            # Count extreme days in the target.
            extreme = int(df[self.target["column"]].sum())

            # Log the split summary.
            self.logger.info(f"{name}: {len(df)} rows, {df['Date'].min().date()} to {df['Date'].max().date()}, {extreme} extreme ({extreme / len(df):.1%}).")

        # Return splits.
        return splits

    @log_step("Store splits")
    def store_splits(self, splits: dict = None) -> dict:
        """
        Store each split to its configured Excel file.

        Args:
            splits (dict): 'train', 'val' and 'test' DataFrames.

        Returns:
            dict: The stored splits.
        """

        # Store each split.
        for name, df in splits.items():

            # Get the output path.
            path = Path(self.output[name])

            # Create the output directory if it does not exist.
            path.parent.mkdir(parents = True, exist_ok = True)

            # Store data.
            df.to_excel(path, index = False)

        # Return splits.
        return splits

    def pipeline(self):
        """
        Run the data split steps in order.
        """

        # Load the feature dataset.
        df = self.load_features(path = self.input["features"])

        # Split in time order.
        splits = self.split_data(df = df)

        # Log each split's summary.
        splits = self.summarize_splits(splits = splits)

        # Store the splits.
        self.store_splits(splits = splits)

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

    # Run the data split.
    DataSplit.run()
