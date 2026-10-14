import copy
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from src.networks.temporal_cnn import TemporalCNNNetwork
from utils.config_loader import ConfigLoader
from utils.error_handler import log_step
from utils.logger import Logger
from utils.windowing import Windowing

class TemporalCNN:
    """
    Train a temporal CNN to predict next-day extreme events from sequences of the previous 10 days, with early stopping on the validation set.
    """

    def __init__(self):
        """
        Create the logger and load the temporal CNN configs.
        """

        # Create a logger named after the class.
        self.logger = Logger().get_logger(self.__class__.__name__)

        # Load configs.
        self.config = ConfigLoader().load("config-temporal-cnn")

        # Get the input configs.
        self.input = self.config["input"]

        # Get the feature columns.
        self.features = self.config["features"]

        # Get the target configs.
        self.target = self.config["target"]

        # Get the window length.
        self.window = self.config["window"]

        # Get the network configs.
        self.network_config = self.config["network"]

        # Get the training configs.
        self.training = self.config["training"]

        # Get the output configs.
        self.output = self.config["output"]

        # Start without a fitted scaler.
        self.scaler = None

    @log_step("Load splits")
    def load_splits(self) -> dict:
        """
        Load the train, val and test sets stored by the data split.

        Returns:
            dict: 'train', 'val' and 'test' DataFrames.
        """

        # Load each split.
        return {name: pd.read_excel(self.input[name]) for name in ("train", "val", "test")}

    @log_step("Feature scaling")
    def scale_features(self, splits: dict = None) -> dict:
        """
        Scale each feature to mean 0 and standard deviation 1, using statistics from the training set only.

        Args:
            splits (dict): 'train', 'val' and 'test' DataFrames.

        Returns:
            dict: The splits with scaled feature columns.
        """

        # Fit the scaler on the training features only.
        self.scaler = StandardScaler().fit(splits["train"][self.features])

        # Scale a copy of each split.
        scaled = {name: df.copy() for name, df in splits.items()}

        # Apply the training scaling to every split.
        for df in scaled.values():

            # Replace the features with their scaled values.
            df[self.features] = self.scaler.transform(df[self.features])

        # Return scaled splits.
        return scaled

    @log_step("Build windows")
    def build_windows(self, splits: dict = None) -> dict:
        """
        Turn each split into 10-day sequences shaped [samples, features, days] for the convolution layers.

        Args:
            splits (dict): Scaled 'train', 'val' and 'test' DataFrames.

        Returns:
            dict: For each split, 'X' (samples, features, days), 'y' and 'dates'.
        """

        # Build the windows.
        windows = Windowing(window = self.window).build(splits = splits, features = self.features, target = self.target["column"])

        # Reorder each split for the convolutions.
        for name, data in windows.items():

            # Move features before days and use 32-bit floats.
            data["X"] = np.transpose(data["X"], (0, 2, 1)).astype(np.float32)

            # Log the split shape.
            self.logger.info(f"{name}: X {data['X'].shape}, y {data['y'].shape}.")

        # Return windows.
        return windows

    @log_step("Build network")
    def build_network(self) -> TemporalCNNNetwork:
        """
        Fix the random seeds and create the network from the configs.

        Returns:
            TemporalCNNNetwork: Untrained network.
        """

        # Fix the seed for weight initialisation, dropout and shuffling.
        torch.manual_seed(self.training["seed"])

        # Create the network.
        return TemporalCNNNetwork(num_features = len(self.features), window = self.window, **self.network_config)

    @log_step("Model training")
    def train_model(self, network: TemporalCNNNetwork = None, windows: dict = None) -> dict:
        """
        Train with Adam and cross-entropy, stop when validation loss stops improving, and restore the best epoch's weights.

        Args:
            network (TemporalCNNNetwork): Untrained network.
            windows (dict): Windows for 'train' and 'val'.

        Returns:
            dict: The trained 'network', its 'best_epoch', 'best_val_loss' and 'epochs_run'.
        """

        # Batch the training windows in a reproducible shuffled order.
        train_loader = DataLoader(TensorDataset(torch.tensor(windows["train"]["X"]), torch.tensor(windows["train"]["y"], dtype = torch.long)), batch_size = self.training["batch_size"], shuffle = True, generator = torch.Generator().manual_seed(self.training["seed"]))

        # Get the validation windows.
        X_val = torch.tensor(windows["val"]["X"])

        # Get the validation labels.
        y_val = torch.tensor(windows["val"]["y"], dtype = torch.long)

        # Weight the classes by inverse frequency if configured, otherwise treat them equally.
        weights = torch.tensor(len(windows["train"]["y"]) / (2 * np.bincount(windows["train"]["y"], minlength = 2)), dtype = torch.float32) if self.training["class_weights"] else None

        # Create the cross-entropy loss.
        loss_fn = nn.CrossEntropyLoss(weight = weights)

        # Create the Adam optimizer.
        optimizer = torch.optim.Adam(network.parameters(), lr = self.training["learning_rate"])

        # Start with no best epoch.
        best = {"val_loss": float("inf"), "epoch": 0, "state": None}

        # Count epochs without improvement.
        stale = 0

        # Train epoch by epoch.
        for epoch in range(1, self.training["max_epochs"] + 1):

            # Switch on dropout.
            network.train()

            # Start the epoch's training loss.
            train_loss = 0.0

            # Learn from each batch.
            for X, y in train_loader:

                # Reset gradients.
                optimizer.zero_grad()

                # Score the batch and calculate the loss.
                loss = loss_fn(network(X), y)

                # Calculate gradients.
                loss.backward()

                # Update the weights.
                optimizer.step()

                # Add the batch loss, weighted by batch size.
                train_loss += loss.item() * len(y)

            # Switch off dropout.
            network.eval()

            # Calculate the validation loss without gradients.
            with torch.no_grad():

                # Score the validation windows.
                val_loss = loss_fn(network(X_val), y_val).item()

            # Log the epoch's losses.
            self.logger.debug(f"Epoch {epoch}: train loss {train_loss / len(train_loader.dataset):.4f}, val loss {val_loss:.4f}.")

            # Keep the weights if validation loss improved.
            if val_loss < best["val_loss"]:

                # Store the new best epoch.
                best = {"val_loss": val_loss, "epoch": epoch, "state": copy.deepcopy(network.state_dict())}

                # Reset the count.
                stale = 0

            # Otherwise count an epoch without improvement.
            else:

                # Increase the count.
                stale += 1

            # Stop after patience epochs without improvement.
            if stale >= self.training["patience"]:

                # Leave the loop.
                break

        # Restore the best epoch's weights.
        network.load_state_dict(best["state"])

        # Log the stopping point.
        self.logger.info(f"Stopped after {epoch} epochs, best epoch {best['epoch']} with val loss {best['val_loss']:.4f}.")

        # Return the trained network.
        return {"network": network, "best_epoch": best["epoch"], "best_val_loss": best["val_loss"], "epochs_run": epoch}

    @log_step("Train and val metrics")
    def score_model(self, result: dict = None, windows: dict = None) -> dict:
        """
        Score the trained network on the train and val windows. Test is left for model evaluation.

        Args:
            result (dict): The trained 'network' and training details.
            windows (dict): Windows for 'train' and 'val'.

        Returns:
            dict: The result with a 'metrics' entry per split.
        """

        # Start with no metrics.
        result["metrics"] = {}

        # Score each split.
        for name in ("train", "val"):

            # Get the true labels.
            y = windows[name]["y"]

            # Predict the class with the highest probability.
            predictions = result["network"].predict_proba(torch.tensor(windows[name]["X"])).argmax(dim = 1).numpy()

            # Calculate the metrics for the extreme class.
            metrics = {
                "accuracy": accuracy_score(y, predictions),
                "precision": precision_score(y, predictions, zero_division = 0),
                "recall": recall_score(y, predictions, zero_division = 0),
                "f1": f1_score(y, predictions, zero_division = 0),
                "confusion": confusion_matrix(y, predictions, labels = [0, 1]).tolist(),
            }

            # Store the split's metrics.
            result["metrics"][name] = metrics

            # Log the split's metrics.
            self.logger.info(f"{name}: accuracy {metrics['accuracy']:.2f}, precision {metrics['precision']:.2f}, recall {metrics['recall']:.2f}, F1 {metrics['f1']:.2f}, confusion {metrics['confusion']}.")

        # Return the result with metrics.
        return result

    @log_step("Save model")
    def save_model(self, result: dict = None, path: str = None) -> dict:
        """
        Save the network weights with the scaler, configs and metrics, so evaluation can rebuild the network and its inputs.

        Args:
            result (dict): The trained 'network', training details and 'metrics'.
            path (str): Output file path.

        Returns:
            dict: The saved bundle.
        """

        # Bundle the weights with what is needed to use them.
        bundle = {
            "state_dict": result["network"].state_dict(),
            "network": self.network_config,
            "scaler": self.scaler,
            "features": self.features,
            "window": self.window,
            "training": self.training,
            "best_epoch": result["best_epoch"],
            "epochs_run": result["epochs_run"],
            "metrics": result["metrics"],
        }

        # Create the output directory if it does not exist.
        Path(path).parent.mkdir(parents = True, exist_ok = True)

        # Save the bundle.
        joblib.dump(bundle, path)

        # Return the bundle.
        return bundle

    def pipeline(self):
        """
        Run the temporal CNN steps in order.
        """

        # Load the splits.
        splits = self.load_splits()

        # Scale features using the training set.
        splits = self.scale_features(splits = splits)

        # Build 10-day sequences.
        windows = self.build_windows(splits = splits)

        # Create the network.
        network = self.build_network()

        # Train with early stopping.
        result = self.train_model(network = network, windows = windows)

        # Score on train and val.
        result = self.score_model(result = result, windows = windows)

        # Save the trained network.
        self.save_model(result = result, path = self.output["model"])

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

    # Run the temporal CNN.
    TemporalCNN.run()
