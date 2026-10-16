import copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from src.models.networks.temporal_cnn import TemporalCNNNetwork
from utils.windowing import Windowing

class TemporalCNNModel:
    """
    Temporal CNN on scaled 10-day sequences, trained with early stopping on the validation set.
    Its settings, including which network to train, come from the config.
    """

    # Networks the config can choose by name.
    NETWORK_CLASSES = {"TemporalCNNNetwork": TemporalCNNNetwork}

    def __init__(self, features: list = None, target: str = None, window: int = 10, network: str = "TemporalCNNNetwork", channels: list = None, kernel_size: int = 3,
                 dropout: float = 0.3, seed: int = 42, batch_size: int = 32, learning_rate: float = 0.001, max_epochs: int = 200,
                 patience: int = 15, class_weights: bool = False, weight_decay: float = 0.0, early_stopping_metric: str = "loss",
                 tune_threshold: bool = False, threshold_metric: str = "f1"):
        """
        Store the model settings.

        Args:
            features (list): Feature columns used in each window.
            target (str): Target column name.
            window (int): Number of days per window.
            network (str): Name of the network class to train, from NETWORK_CLASSES.
            channels (list): Output channels of the two convolution layers, e.g. [32, 64].
            kernel_size (int): Number of days each convolution filter looks at.
            dropout (float): Share of activations dropped during training.
            seed (int): Seed for weights, dropout and batch shuffling.
            batch_size (int): Windows per training batch.
            learning_rate (float): Adam learning rate.
            max_epochs (int): Maximum number of training epochs.
            patience (int): Epochs without improvement before stopping.
            class_weights (bool): Weight the loss by inverse class frequency.
            weight_decay (float): Adam weight decay, penalising large weights.
            early_stopping_metric (str): 'loss' (lower val loss) or 'f1' (higher val F1).
            tune_threshold (bool): Pick the decision threshold with the best val score instead of 0.5.
            threshold_metric (str): Val score used to pick the threshold: 'f1' or 'mcc' (MCC cannot be won by flagging every day).
        """

        # Store the feature columns.
        self.features = features

        # Store the target column.
        self.target = target

        # Store the window length.
        self.window = window

        # Store the network class name.
        self.network_name = network

        # Store the network settings.
        self.network_config = {"channels": channels or [32, 64], "kernel_size": kernel_size, "dropout": dropout}

        # Store the training settings.
        self.training = {"seed": seed, "batch_size": batch_size, "learning_rate": learning_rate, "max_epochs": max_epochs, "patience": patience,
                         "class_weights": class_weights, "weight_decay": weight_decay, "early_stopping_metric": early_stopping_metric, "tune_threshold": tune_threshold, "threshold_metric": threshold_metric}

        # Call a day extreme when its probability reaches 0.5 unless tuned.
        self.threshold = 0.5

        # Start without a fitted scaler.
        self.scaler = None

        # Start without a trained network.
        self.network = None

        # Start without a training history.
        self.history = []

        # Start without a best epoch.
        self.best_epoch = None

    def windows(self, splits: dict = None) -> dict:
        """
        Scale features with the train-fitted scaler and build 10-day sequences shaped [samples, features, days].

        Args:
            splits (dict): 'train', 'val' and 'test' DataFrames.

        Returns:
            dict: For each split, 'X' (samples, features, days), 'y' and 'dates'.
        """

        # Scale a copy of each split.
        scaled = {name: df.copy() for name, df in splits.items()}

        # Apply the training scaling to every split.
        for df in scaled.values():

            # Replace the features with their scaled values.
            df[self.features] = self.scaler.transform(df[self.features])

        # Build the windows.
        windows = Windowing(window = self.window).build(splits = scaled, features = self.features, target = self.target)

        # Reorder each split for the convolutions.
        for data in windows.values():

            # Move features before days and use 32-bit floats.
            data["X"] = np.transpose(data["X"], (0, 2, 1)).astype(np.float32)

        # Return windows.
        return windows

    def fit(self, splits: dict = None) -> "TemporalCNNModel":
        """
        Fit the scaler on train, train with Adam and cross-entropy, stop when the val metric stops improving,
        restore the best epoch's weights and, if configured, tune the threshold on val.

        Args:
            splits (dict): 'train', 'val' and 'test' DataFrames.

        Returns:
            TemporalCNNModel: The fitted model.
        """

        # Fit the scaler on the training features only.
        self.scaler = StandardScaler().fit(splits["train"][self.features])

        # Build the windows.
        windows = self.windows(splits = splits)

        # Fix the seed for weight initialisation, dropout and shuffling.
        torch.manual_seed(self.training["seed"])

        # Create the configured network.
        self.network = self.NETWORK_CLASSES[self.network_name](num_features = len(self.features), window = self.window, **self.network_config)

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
        optimizer = torch.optim.Adam(self.network.parameters(), lr = self.training["learning_rate"], weight_decay = self.training["weight_decay"])

        # Start with no best epoch.
        best = {"score": -float("inf"), "epoch": 0, "state": None}

        # Count epochs without improvement.
        stale = 0

        # Train epoch by epoch.
        for epoch in range(1, self.training["max_epochs"] + 1):

            # Switch on dropout.
            self.network.train()

            # Start the epoch's training loss.
            train_loss = 0.0

            # Learn from each batch.
            for X, y in train_loader:

                # Reset gradients.
                optimizer.zero_grad()

                # Score the batch and calculate the loss.
                loss = loss_fn(self.network(X), y)

                # Calculate gradients.
                loss.backward()

                # Update the weights.
                optimizer.step()

                # Add the batch loss, weighted by batch size.
                train_loss += loss.item() * len(y)

            # Switch off dropout.
            self.network.eval()

            # Score the validation windows without gradients.
            with torch.no_grad():

                # Get the validation class scores.
                val_scores = self.network(X_val)

            # Calculate the validation loss.
            val_loss = loss_fn(val_scores, y_val).item()

            # Calculate the validation F1 for the extreme class.
            val_f1 = f1_score(y_val.numpy(), val_scores.argmax(dim = 1).numpy(), zero_division = 0)

            # Record the epoch.
            self.history.append({"epoch": epoch, "train_loss": train_loss / len(train_loader.dataset), "val_loss": val_loss, "val_f1": val_f1})

            # Score the epoch by the configured metric, higher being better.
            score = val_f1 if self.training["early_stopping_metric"] == "f1" else -val_loss

            # Keep the weights if the score improved.
            if score > best["score"]:

                # Store the new best epoch.
                best = {"score": score, "epoch": epoch, "state": copy.deepcopy(self.network.state_dict())}

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
        self.network.load_state_dict(best["state"])

        # Store the best epoch.
        self.best_epoch = best["epoch"]

        # Tune the threshold if configured.
        if self.training["tune_threshold"]:

            # Predict the probability of an extreme day for each validation window.
            probabilities = self.network.predict_proba(X_val)[:, 1].numpy()

            # Try thresholds from 0.05 to 0.95.
            thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)

            # Get the configured scoring function.
            metric = matthews_corrcoef if self.training["threshold_metric"] == "mcc" else lambda y, pred: f1_score(y, pred, zero_division = 0)

            # Score each threshold on the validation windows.
            scores = [metric(windows["val"]["y"], (probabilities >= t).astype(int)) for t in thresholds]

            # Keep the threshold with the best score.
            self.threshold = float(thresholds[int(np.argmax(scores))])

        # Return the fitted model.
        return self

    def predict_proba(self, splits: dict = None) -> dict:
        """
        Predict the probability of an extreme day for every window of every split.

        Args:
            splits (dict): 'train', 'val' and 'test' DataFrames.

        Returns:
            dict: For each split, true labels 'y', probabilities 'proba' and 'dates'.
        """

        # Build the windows.
        windows = self.windows(splits = splits)

        # Predict the extreme class probability per split.
        return {name: {"y": data["y"], "proba": self.network.predict_proba(torch.tensor(data["X"]))[:, 1].numpy(), "dates": data["dates"]} for name, data in windows.items()}

    def predict(self, splits: dict = None) -> dict:
        """
        Predict extreme (1) or not (0) for every window of every split, using the model's threshold.

        Args:
            splits (dict): 'train', 'val' and 'test' DataFrames.

        Returns:
            dict: For each split, true labels 'y', probabilities 'proba', predictions 'pred' and 'dates'.
        """

        # Predict probabilities.
        predictions = self.predict_proba(splits = splits)

        # Apply the threshold to each split.
        for data in predictions.values():

            # Call a day extreme when its probability reaches the threshold.
            data["pred"] = (data["proba"] >= self.threshold).astype(int)

        # Return predictions.
        return predictions
