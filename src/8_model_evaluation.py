from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from src.networks.temporal_cnn import TemporalCNNNetwork
from utils.config_loader import ConfigLoader
from utils.error_handler import log_step
from utils.logger import Logger
from utils.windowing import Windowing

class ModelEvaluation:
    """
    Score the saved models on train, val and test, compare them on the test set with simple baselines, and check for overfitting.
    This is the only step that scores on the test set; no model is trained or changed here.
    """

    def __init__(self):
        """
        Create the logger and load the model evaluation configs.
        """

        # Create a logger named after the class.
        self.logger = Logger().get_logger(self.__class__.__name__)

        # Load configs.
        self.config = ConfigLoader().load("config-evaluation")

        # Get the input configs.
        self.input = self.config["input"]

        # Get the target configs.
        self.target = self.config["target"]

        # Get the saved model paths.
        self.models = self.config["models"]

        # Get the extreme events configs.
        self.extreme_events = self.config["extreme_events"]

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

    @log_step("Load models")
    def load_models(self) -> dict:
        """
        Load every saved model bundle.

        Returns:
            dict: Model name to saved bundle.
        """

        # Load each model.
        return {name: joblib.load(path) for name, path in self.models.items()}

    def predict_random_forest(self, bundle: dict = None, splits: dict = None) -> dict:
        """
        Rebuild the random forest's flattened 10-day windows and predict on every split.

        Args:
            bundle (dict): Saved random forest bundle.
            splits (dict): 'train', 'val' and 'test' DataFrames.

        Returns:
            dict: For each split, true labels 'y' and predictions 'pred'.
        """

        # Build the windows the random forest was trained on.
        windows = Windowing(window = bundle["window"]).build(splits = splits, features = bundle["features"], target = self.target["column"])

        # Predict on each split's flattened windows.
        return {name: {"y": data["y"], "pred": bundle["model"].predict(data["X"].reshape(len(data["X"]), -1))} for name, data in windows.items()}

    def predict_temporal_cnn(self, bundle: dict = None, splits: dict = None) -> dict:
        """
        Rebuild the temporal CNN with its saved weights, scale with its saved scaler, and predict on every split with its saved threshold.

        Args:
            bundle (dict): Saved temporal CNN bundle.
            splits (dict): 'train', 'val' and 'test' DataFrames.

        Returns:
            dict: For each split, true labels 'y' and predictions 'pred'.
        """

        # Rebuild the network.
        network = TemporalCNNNetwork(num_features = len(bundle["features"]), window = bundle["window"], **bundle["network"])

        # Load the saved weights.
        network.load_state_dict(bundle["state_dict"])

        # Scale a copy of each split with the saved, train-fitted scaler.
        scaled = {name: df.copy() for name, df in splits.items()}

        # Apply the scaling to every split.
        for df in scaled.values():

            # Replace the features with their scaled values.
            df[bundle["features"]] = bundle["scaler"].transform(df[bundle["features"]])

        # Build the windows the network was trained on.
        windows = Windowing(window = bundle["window"]).build(splits = scaled, features = bundle["features"], target = self.target["column"])

        # Start with no predictions.
        predictions = {}

        # Predict on each split.
        for name, data in windows.items():

            # Reorder windows to [samples, features, days].
            x = torch.tensor(np.transpose(data["X"], (0, 2, 1)), dtype = torch.float32)

            # Call a day extreme when its probability reaches the saved threshold.
            predictions[name] = {"y": data["y"], "pred": (network.predict_proba(x)[:, 1].numpy() >= bundle["threshold"]).astype(int)}

        # Return predictions.
        return predictions

    @log_step("Model predictions")
    def predict_models(self, models: dict = None, splits: dict = None) -> dict:
        """
        Predict with every saved model on every split.

        Args:
            models (dict): Model name to saved bundle.
            splits (dict): 'train', 'val' and 'test' DataFrames.

        Returns:
            dict: Model name to its predictions per split.
        """

        # Use the temporal CNN path for bundles with network weights, and the random forest path otherwise.
        return {name: self.predict_temporal_cnn(bundle = bundle, splits = splits) if "state_dict" in bundle else self.predict_random_forest(bundle = bundle, splits = splits) for name, bundle in models.items()}

    @log_step("Baseline predictions")
    def predict_baselines(self, splits: dict = None) -> dict:
        """
        Predict on the test set with three rules that need no learning: never extreme, always extreme, and persistence
        (tomorrow is extreme if today's move was beyond the threshold).

        Args:
            splits (dict): 'train', 'val' and 'test' DataFrames.

        Returns:
            dict: Baseline name to its test predictions.
        """

        # Get the test labels.
        y = splits["test"][self.target["column"]].to_numpy()

        # Flag test days whose own move was beyond the threshold.
        today = (splits["test"]["daily_return"].abs() > self.extreme_events["threshold"]).astype(int).to_numpy()

        # Return the three rules' predictions.
        return {
            "never_extreme": {"test": {"y": y, "pred": np.zeros_like(y)}},
            "always_extreme": {"test": {"y": y, "pred": np.ones_like(y)}},
            "persistence": {"test": {"y": y, "pred": today}},
        }

    @log_step("Scoring")
    def score(self, predictions: dict = None) -> pd.DataFrame:
        """
        Score every model and baseline on every split it has predictions for.

        Args:
            predictions (dict): Model or baseline name to its predictions per split.

        Returns:
            pd.DataFrame: One row per model and split with accuracy, precision, recall, F1 and the confusion matrix.
        """

        # Start with no rows.
        rows = []

        # Score each model and split.
        for model, splits in predictions.items():

            # Score each split.
            for split, data in splits.items():

                # Add the split's metrics for the extreme class.
                rows.append({
                    "model": model,
                    "split": split,
                    "samples": len(data["y"]),
                    "accuracy": accuracy_score(data["y"], data["pred"]),
                    "precision": precision_score(data["y"], data["pred"], zero_division = 0),
                    "recall": recall_score(data["y"], data["pred"], zero_division = 0),
                    "f1": f1_score(data["y"], data["pred"], zero_division = 0),
                    "confusion": confusion_matrix(data["y"], data["pred"], labels = [0, 1]).tolist(),
                })

        # Return the table.
        return pd.DataFrame(rows)

    @log_step("Report")
    def report(self, table: pd.DataFrame = None) -> pd.DataFrame:
        """
        Log the test-set comparison, best first, and each trained model's train, val and test F1 to check for overfitting.

        Args:
            table (pd.DataFrame): One row per model and split.

        Returns:
            pd.DataFrame: The same table.
        """

        # Log the test-set comparison, best F1 first.
        for row in table[table["split"] == "test"].sort_values("f1", ascending = False).itertuples():

            # Log the model's test metrics.
            self.logger.info(f"test  {row.model:<22} accuracy {row.accuracy:.3f}, precision {row.precision:.3f}, recall {row.recall:.3f}, F1 {row.f1:.3f}, confusion {row.confusion}.")

        # Log each trained model's F1 across splits.
        for model in self.models:

            # Get the model's F1 per split.
            f1 = table[table["model"] == model].set_index("split")["f1"]

            # Log the overfitting check.
            self.logger.info(f"F1 by split  {model:<22} train {f1['train']:.3f}, val {f1['val']:.3f}, test {f1['test']:.3f}.")

        # Return the table.
        return table

    @log_step("Save report")
    def save_report(self, table: pd.DataFrame = None, path: str = None) -> pd.DataFrame:
        """
        Store the full evaluation table to CSV.

        Args:
            table (pd.DataFrame): One row per model and split.
            path (str): Output file path.

        Returns:
            pd.DataFrame: The stored table.
        """

        # Create the output directory if it does not exist.
        Path(path).parent.mkdir(parents = True, exist_ok = True)

        # Store the table.
        table.to_csv(path, index = False)

        # Return the table.
        return table

    def pipeline(self):
        """
        Run the model evaluation steps in order.
        """

        # Load the splits.
        splits = self.load_splits()

        # Load the saved models.
        models = self.load_models()

        # Predict with the models on every split.
        predictions = self.predict_models(models = models, splits = splits)

        # Add the baselines' test predictions.
        predictions.update(self.predict_baselines(splits = splits))

        # Score everything.
        table = self.score(predictions = predictions)

        # Log the comparison and overfitting check.
        table = self.report(table = table)

        # Store the table.
        self.save_report(table = table, path = self.output["report"])

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

    # Run the model evaluation.
    ModelEvaluation.run()
