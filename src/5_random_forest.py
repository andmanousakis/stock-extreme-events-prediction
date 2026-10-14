from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid
from utils.config_loader import ConfigLoader
from utils.error_handler import log_step
from utils.logger import Logger
from utils.windowing import Windowing

class RandomForest:
    """
    Train a random forest to predict next-day extreme events from the previous 10 days, tuned on the validation set.
    """

    def __init__(self):
        """
        Create the logger and load the random forest configs.
        """

        # Create a logger named after the class.
        self.logger = Logger().get_logger(self.__class__.__name__)

        # Load configs.
        self.config = ConfigLoader().load("config-random-forest")

        # Get the input configs.
        self.input = self.config["input"]

        # Get the feature columns.
        self.features = self.config["features"]

        # Get the target configs.
        self.target = self.config["target"]

        # Get the window length.
        self.window = self.config["window"]

        # Get the model configs.
        self.model_config = self.config["model"]

        # Get the output configs.
        self.output = self.config["output"]

    @log_step("Load splits")
    def load_splits(self) -> dict:
        """
        Load the train, val and test sets stored by the data split.

        Returns:
            dict: 'train', 'val' and 'test' DataFrames.
        """

        # Load each split.
        return {name: pd.read_excel(self.input[name]) for name in ("train", "val", "test")}

    @log_step("Build windows")
    def build_windows(self, splits: dict = None) -> dict:
        """
        Turn each split into 10-day windows, flattened to one row of window x features columns per sample.

        Args:
            splits (dict): 'train', 'val' and 'test' DataFrames.

        Returns:
            dict: For each split, 'X' (samples, window * features), 'y' and 'dates'.
        """

        # Build the windows.
        windows = Windowing(window = self.window).build(splits = splits, features = self.features, target = self.target["column"])

        # Flatten each window to one row.
        for name, data in windows.items():

            # Flatten days and features into one row per sample.
            data["X"] = data["X"].reshape(len(data["X"]), -1)

            # Log the split shape.
            self.logger.info(f"{name}: X {data['X'].shape}, y {data['y'].shape}.")

        # Return windows.
        return windows

    @log_step("Hyperparameter tuning")
    def tune_model(self, windows: dict = None) -> dict:
        """
        Train one random forest per parameter combination on train, score each by F1 on val, and keep the best.

        Args:
            windows (dict): Flattened windows for 'train' and 'val'.

        Returns:
            dict: The best 'model', its 'params' and its validation 'f1'.
        """

        # Start with no best model.
        best = {"model": None, "params": None, "f1": -1.0}

        # Try every parameter combination.
        for params in ParameterGrid(self.model_config["param_grid"]):

            # Train on the training windows.
            model = RandomForestClassifier(**params, random_state = self.model_config["random_state"], n_jobs = -1).fit(windows["train"]["X"], windows["train"]["y"])

            # Score F1 for the extreme class on the validation windows.
            f1 = f1_score(windows["val"]["y"], model.predict(windows["val"]["X"]), zero_division = 0)

            # Log the combination's score.
            self.logger.debug(f"{params}: val F1 {f1:.3f}.")

            # Keep the combination if it is the best so far.
            if f1 > best["f1"]:

                # Store the new best.
                best = {"model": model, "params": params, "f1": f1}

        # Log the best combination.
        self.logger.info(f"Best params {best['params']}, val F1 {best['f1']:.3f}.")

        # Return the best model.
        return best

    @log_step("Save model")
    def save_model(self, best: dict = None, path: str = None) -> dict:
        """
        Save the best model with its parameters, features and window length so evaluation can rebuild the same inputs.

        Args:
            best (dict): The best 'model', 'params' and 'f1'.
            path (str): Output file path.

        Returns:
            dict: The saved bundle.
        """

        # Bundle the model with what is needed to use it.
        bundle = {"model": best["model"], "params": best["params"], "val_f1": best["f1"], "features": self.features, "window": self.window}

        # Create the output directory if it does not exist.
        Path(path).parent.mkdir(parents = True, exist_ok = True)

        # Save the bundle.
        joblib.dump(bundle, path)

        # Return the bundle.
        return bundle

    def pipeline(self):
        """
        Run the random forest steps in order.
        """

        # Load the splits.
        splits = self.load_splits()

        # Build 10-day windows.
        windows = self.build_windows(splits = splits)

        # Tune on the validation set.
        best = self.tune_model(windows = windows)

        # Save the best model.
        self.save_model(best = best, path = self.output["model"])

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

    # Run the random forest.
    RandomForest.run()
