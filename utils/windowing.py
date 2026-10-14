import numpy as np
import pandas as pd

class Windowing:
    """
    Build fixed-length windows of past days for time series models.
    """

    def __init__(self, window: int = 10):
        """
        Set the number of days in each window.

        Args:
            window (int): Number of consecutive days per window, ending on the day being predicted from.
        """

        # Store the window length.
        self.window = window

    def build(self, splits: dict = None, features: list = None, target: str = None) -> dict:
        """
        Build windows for each split. Windows may look back into earlier splits for their input days,
        but each window belongs to the split of its last day, so targets never cross splits.

        Args:
            splits (dict): 'train', 'val' and 'test' DataFrames in date order, each with a 'Date' column.
            features (list): Feature column names.
            target (str): Target column name.

        Returns:
            dict: For each split, 'X' (samples, window, features), 'y' (samples,) and 'dates' (last day of each window).
        """

        # Join the splits in date order, remembering which split each row came from.
        df = pd.concat([part.assign(_split = name) for name, part in splits.items()], ignore_index = True)

        # Get the feature values.
        values = df[features].to_numpy(dtype = float)

        # Get the window end positions that have a full window of past days.
        ends = np.arange(self.window - 1, len(df))

        # Stack each window of past days.
        X = np.stack([values[end - self.window + 1:end + 1] for end in ends])

        # Get the target of each window's last day.
        y = df[target].to_numpy()[ends]

        # Get the date of each window's last day.
        dates = df["Date"].to_numpy()[ends]

        # Get the split of each window's last day.
        owners = df["_split"].to_numpy()[ends]

        # Return the windows grouped by split.
        return {name: {"X": X[owners == name], "y": y[owners == name], "dates": dates[owners == name]} for name in splits}
