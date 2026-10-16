import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import ParameterGrid
from utils.windowing import Windowing

class RandomForestModel:
    """
    Random forest on flattened 10-day windows, tuned on the validation set by F1.
    """

    def __init__(self, features: list = None, target: str = None, window: int = 10, param_grid: dict = None, random_state: int = 42):
        """
        Store the model settings.

        Args:
            features (list): Feature columns used in each window.
            target (str): Target column name.
            window (int): Number of days per window.
            param_grid (dict): Random forest settings to try, e.g. {'max_depth': [3, 5]}.
            random_state (int): Seed for reproducible trees.
        """

        # Store the feature columns.
        self.features = features

        # Store the target column.
        self.target = target

        # Store the window length.
        self.window = window

        # Store the settings to try.
        self.param_grid = param_grid or {}

        # Store the seed.
        self.random_state = random_state

        # Call a day extreme when its probability reaches 0.5.
        self.threshold = 0.5

        # Start without a trained forest.
        self.model = None

        # Start without chosen settings.
        self.best_params = None

        # Start without tuning results.
        self.tuning_results = []

        # Start without a training curve.
        self.history = []

    def windows(self, splits: dict = None) -> dict:
        """
        Build 10-day windows for each split, flattened to one row of window x features columns.

        Args:
            splits (dict): 'train', 'val' and 'test' DataFrames.

        Returns:
            dict: For each split, 'X' (samples, window * features), 'y' and 'dates'.
        """

        # Build the windows.
        windows = Windowing(window = self.window).build(splits = splits, features = self.features, target = self.target)

        # Flatten each split's windows.
        for data in windows.values():

            # Flatten days and features into one row per sample.
            data["X"] = data["X"].reshape(len(data["X"]), -1)

        # Return windows.
        return windows

    def fit(self, splits: dict = None) -> "RandomForestModel":
        """
        Train one forest per setting combination on train, score each by F1 on val, and keep the best.

        Args:
            splits (dict): 'train', 'val' and 'test' DataFrames.

        Returns:
            RandomForestModel: The fitted model.
        """

        # Build the windows.
        windows = self.windows(splits = splits)

        # Start with no best F1.
        best_f1 = -1.0

        # Try every setting combination.
        for params in ParameterGrid(self.param_grid):

            # Train on the training windows.
            model = RandomForestClassifier(**params, random_state = self.random_state, n_jobs = -1).fit(windows["train"]["X"], windows["train"]["y"])

            # Score F1 for the extreme class on the validation windows.
            f1 = f1_score(windows["val"]["y"], model.predict(windows["val"]["X"]), zero_division = 0)

            # Record the combination's score.
            self.tuning_results.append({"params": params, "val_f1": f1})

            # Keep the combination if it is the best so far.
            if f1 > best_f1:

                # Store the new best.
                best_f1, self.model, self.best_params = f1, model, params

        # Record how the error moves as trees are added to the chosen forest.
        self.history = self.training_curve(windows = windows)

        # Return the fitted model.
        return self

    def training_curve(self, windows: dict = None) -> list:
        """
        Measure the chosen forest using only its first 1, 2, ... N trees: train and val log loss, and val F1.

        Args:
            windows (dict): Flattened windows for 'train' and 'val'.

        Returns:
            list: One entry per number of trees with 'trees', 'train_loss', 'val_loss' and 'val_f1'.
        """

        # Get each tree's extreme class probability on train and val, one row per tree.
        probabilities = {name: np.stack([tree.predict_proba(windows[name]["X"])[:, 1] for tree in self.model.estimators_]) for name in ("train", "val")}

        # Average the first k trees for every k.
        averages = {name: np.cumsum(p, axis = 0) / np.arange(1, len(p) + 1)[:, None] for name, p in probabilities.items()}

        # Score the forest at every size.
        return [{
            "trees": k + 1,
            "train_loss": log_loss(windows["train"]["y"], averages["train"][k], labels = [0, 1]),
            "val_loss": log_loss(windows["val"]["y"], averages["val"][k], labels = [0, 1]),
            "val_f1": f1_score(windows["val"]["y"], (averages["val"][k] >= self.threshold).astype(int), zero_division = 0),
        } for k in range(len(self.model.estimators_))]

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
        return {name: {"y": data["y"], "proba": self.model.predict_proba(data["X"])[:, 1], "dates": data["dates"]} for name, data in windows.items()}

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
