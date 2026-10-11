import inspect
import pandas as pd
from rich import print
from utils.config_loader import ConfigLoader

class DataPreprocessing:
    """
    Load raw stock price data, clean it, label extreme events and split it for ML training.
    """

    def __init__(self):
        """
        Load the data preprocessing configs.
        """

        # Load configs.
        self.config = ConfigLoader().load("config-preprocessing")

    def class_name(self) -> str:
        """
        Get the name of the class.

        Returns:
            str: Class name.
        """

        # Return class name.
        return self.__class__.__name__

    def method_name(self) -> str:
        """
        Get the name of the method that called this one.

        Returns:
            str: Caller method name.
        """

        # Return caller method name.
        return inspect.currentframe().f_back.f_code.co_name

    def load_raw_data(self, path: str = None) -> pd.DataFrame:
        """
        Load the raw price data stored by DataRetrieval.

        Args:
            path (str): Raw data file path.

        Returns:
            pd.DataFrame: Raw price data with a 'Date' column.
        """

        try:

            # Load raw data.
            df = pd.read_excel(path)

            # Print message.
            print(f"[green]Success[/green] -> {self.class_name()} > {self.method_name()}: Load raw data.")

        except Exception as e:

            # Print message.
            print(f"[red]Error[/red] -> {self.class_name()} > {self.method_name()}: {e}.")

            # Exit.
            raise SystemExit

        # Return data.
        return df

    def expand_ts(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Expand the time series to include every calendar day, adding weekends and holidays as empty rows.

        Args:
            df (pd.DataFrame): Price data with a 'Date' column.

        Returns:
            pd.DataFrame: Data indexed by a continuous daily 'Date' index.
        """

        try:

            # Convert 'Date' to datetime.
            df["Date"] = pd.to_datetime(df["Date"])

            # Set 'Date' as the index.
            df.set_index("Date", inplace = True)

            # Create a complete daily date range.
            full_date_range = pd.date_range(start = df.index.min(), end = df.index.max(), freq = "D")

            # Reindex to include all dates.
            df = df.reindex(full_date_range)

            # Rename index.
            df.index.name = "Date"

            # Print message.
            print(f"[green]Success[/green] -> {self.class_name()} > {self.method_name()}: TS expansion.")

        except Exception as e:

            # Print message.
            print(f"[red]Error[/red] -> {self.class_name()} > {self.method_name()}: {e}.")

            # Exit.
            raise SystemExit

        # Return data.
        return df

    def imputation(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Fill missing values on non-trading days.

        Args:
            df (pd.DataFrame): Daily-expanded price data.

        Returns:
            pd.DataFrame: Data with zero volume and forward-filled prices on non-trading days.
        """

        try:

            # Get the columns to impute.
            volume_column = self.config["imputation"]["volume_column"]

            # Get the price columns to forward-fill.
            price_columns = self.config["imputation"]["price_columns"]

            # Set volume to 0 for non-trading days.
            df[volume_column] = df[volume_column].fillna(0)

            # Forward-fill prices to maintain trends.
            df[price_columns] = df[price_columns].ffill()

            # Print message.
            print(f"[green]Success[/green] -> {self.class_name()} > {self.method_name()}: Missing values imputation.")

        except Exception as e:

            # Print message.
            print(f"[red]Error[/red] -> {self.class_name()} > {self.method_name()}: {e}.")

            # Exit.
            raise SystemExit

        # Return data.
        return df

    def daily_returns(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Calculate daily percentage returns.

        Args:
            df (pd.DataFrame): Price data.

        Returns:
            pd.DataFrame: Data with a 'daily_return' column, first row dropped.
        """

        try:

            # Calculate percentage change.
            df["daily_return"] = df[self.config["daily_returns"]["price_column"]].pct_change() * 100

            # Drop the first row, which has no return.
            df = df.iloc[1:]

            # Print message.
            print(f"[green]Success[/green] -> {self.class_name()} > {self.method_name()}: Daily returns calculation.")

        except Exception as e:

            # Print message.
            print(f"[red]Error[/red] -> {self.class_name()} > {self.method_name()}: {e}.")

            # Exit.
            raise SystemExit

        # Return data.
        return df

    def extreme_events(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Label days whose absolute daily return exceeds the threshold as extreme events.

        Args:
            df (pd.DataFrame): Data with a 'daily_return' column.

        Returns:
            pd.DataFrame: Data with a binary 'extreme_event' column.
        """

        try:

            # Get the extreme event threshold.
            threshold = self.config["extreme_events"]["threshold"]

            # Define extreme events.
            df["extreme_event"] = df["daily_return"].apply(lambda x: 1 if abs(x) > threshold else 0)

            # Print message.
            print(f"[green]Success[/green] -> {self.class_name()} > {self.method_name()}: Extreme events calculation.")

        except Exception as e:

            # Print message.
            print(f"[red]Error[/red] -> {self.class_name()} > {self.method_name()}: {e}.")

            # Exit.
            raise SystemExit

        # Return data.
        return df

    def prep_data_for_ml(self, df: pd.DataFrame = None) -> dict:
        """
        Split the data sequentially into training, validation and test sets and store each set.

        Args:
            df (pd.DataFrame): Fully preprocessed data.

        Returns:
            dict: Features and target for the training, validation and test sets.
        """

        try:

            # Get the split configs.
            split = self.config["split"]

            # Get the output configs.
            output = self.config["output"]

            # Extract features.
            features = df[split["features"]]

            # Extract target.
            target = df[split["target"]]

            # Get the number of samples.
            N = len(df.index)

            # Calculate the training set end index.
            train_end = int(split["train_end"] * N)

            # Calculate the validation set end index.
            val_end = int(split["val_end"] * N)

            # Split features sequentially to preserve time series order.
            X_train, X_val, X_test = features[:train_end], features[train_end:val_end], features[val_end:]

            # Split target sequentially to preserve time series order.
            y_train, y_val, y_test = target[:train_end], target[train_end:val_end], target[val_end:]

            # Store training set.
            df[:train_end].to_excel(output["training_set"], index = True)

            # Store validation set.
            df[train_end:val_end].to_excel(output["validation_set"], index = True)

            # Store test set.
            df[val_end:].to_excel(output["test_set"], index = True)

            # Print message.
            print(f"[green]Success[/green] -> {self.class_name()} > {self.method_name()}: Splitting data.")

        except Exception as e:

            # Print message.
            print(f"[red]Error[/red] -> {self.class_name()} > {self.method_name()}: {e}.")

            # Exit.
            raise SystemExit

        # Return data.
        return {
            "training_data": {
                "features": X_train,
                "target": y_train
            },
            "validation_data": {
                "features": X_val,
                "target": y_val
            },
            "test_data": {
                "features": X_test,
                "target": y_test
            }
        }

    def store_data(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Store the preprocessed dataset to Excel.

        Args:
            df (pd.DataFrame): Preprocessed data.

        Returns:
            pd.DataFrame: The stored data.
        """

        try:

            # Store data.
            df.to_excel(self.config["output"]["dataset"], index = True)

            # Print message.
            print(f"[green]Success[/green] -> {self.class_name()} > {self.method_name()}: Store data.")

        except Exception as e:

            # Print message.
            print(f"[red]Error[/red] -> {self.class_name()} > {self.method_name()}: {e}.")

            # Exit.
            raise SystemExit

        # Return data.
        return df

    def pipeline(self):
        """
        Run the preprocessing steps in order.
        """

        # Load raw data.
        data_df = self.load_raw_data(path = self.config["input"]["raw_data"])

        # Step 1.1.a: Expand TS.
        data_df = self.expand_ts(df = data_df)

        # Step 1.1.b: Imputation.
        data_df = self.imputation(df = data_df)

        # Step 1.2: Calculate daily returns.
        data_df = self.daily_returns(df = data_df)

        # Step 1.3: Define extreme events.
        data_df = self.extreme_events(df = data_df)

        # Step 1.4: Split data into features and target and store them for model training.
        self.prep_data_for_ml(df = data_df)

        # Store data.
        self.store_data(df = data_df)

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
