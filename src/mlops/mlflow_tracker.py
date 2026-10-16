import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
from src.mlops.mlflow_model import MlflowModel

class MlflowTracker:
    """
    Log model runs to MLflow and register the best model with an alias per environment.
    Only active when MLFLOW_TRACKING_URI is set.
    """

    # Metrics logged for every split.
    METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "mcc")

    def __init__(self, experiment: str = None, registered_model: str = None, promote_only_if_better: bool = False):
        """
        Store the tracking settings and read the environment.

        Args:
            experiment (str): MLflow experiment name.
            registered_model (str): Registered model name in the Model Registry.
            promote_only_if_better (bool): Move the environment's alias only if the new version has a higher val F1.
        """

        # Store the experiment name.
        self.experiment = experiment

        # Store the registered model name.
        self.registered_model = registered_model

        # Store the promotion rule.
        self.promote_only_if_better = promote_only_if_better

        # Track only when a tracking server is configured.
        self.enabled = bool(os.environ.get("MLFLOW_TRACKING_URI"))

        # Get the environment the models belong to.
        self.env = os.environ.get("APP_ENV", "dev")

    def log_models(self, models_config: dict = None, models: dict = None, predictions: dict = None, table: pd.DataFrame = None, models_dir: str = None) -> dict:
        """
        Log one run per model with its settings, its metrics on every split, its training and tuning curves,
        its plots and the saved model in MLflow's format.

        Args:
            models_config (dict): Model name to its config entry ('class' and 'params').
            models (dict): Model name to trained model.
            predictions (dict): Model name to its predictions per split ('y', 'proba', 'pred').
            table (pd.DataFrame): One row per model and split with the metrics.
            models_dir (str): Folder holding each model's saved <name>.joblib.

        Returns:
            dict: Model name to its logged model URI.
        """

        # Use the configured experiment.
        mlflow.set_experiment(self.experiment)

        # Start with no logged models.
        uris = {}

        # Log each model as its own run.
        for name, entry in models_config.items():

            # Start the model's run.
            with mlflow.start_run(run_name = f"{self.env}-{name}"):

                # Tag the run with where it came from.
                mlflow.set_tags({"environment": self.env, "model": name, "branch": os.environ.get("GITHUB_REF_NAME", "local"), "commit": os.environ.get("GITHUB_SHA", "local")})

                # Log the model's class and settings, writing lists and dicts as JSON.
                mlflow.log_params({"class": entry["class"], **{key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in entry["params"].items()}})

                # Log every metric of every split, e.g. val_f1.
                for row in table[table["model"] == name].itertuples():

                    # Log the split's metrics, skipping missing values.
                    mlflow.log_metrics({f"{row.split}_{metric}": getattr(row, metric) for metric in self.METRICS if not pd.isna(getattr(row, metric))})

                # Log how the error moved during training and tuning.
                self.log_curves(model = models[name])

                # Log the confusion matrices, ROC and precision-recall curves.
                self.log_plots(predictions = predictions[name])

                # Log the saved model in MLflow's standard format. Loading it needs this project's code installed, as in its images.
                info = mlflow.pyfunc.log_model(name = "model", python_model = MlflowModel(), artifacts = {"model": str(Path(models_dir) / f"{name}.joblib")})

                # Store the logged model URI.
                uris[name] = info.model_uri

        # Return the logged model URIs.
        return uris

    def log_curves(self, model = None):
        """
        Log the model's training curve by step: per epoch for the temporal CNN, per number of trees for the random forest,
        plus the random forest's val F1 for each tuned setting.

        Args:
            model (RandomForestModel | TemporalCNNModel): Trained model with a 'history' and, for the forest, 'tuning_results'.
        """

        # Log each point of the training curve.
        for point in model.history:

            # Use the epoch or the number of trees as the step.
            step = point.get("epoch", point.get("trees"))

            # Log the point's losses and F1.
            mlflow.log_metrics({f"training/{key}": value for key, value in point.items() if key not in ("epoch", "trees")}, step = step)

        # Log the val F1 of each tuned setting, in the order they were tried.
        for step, result in enumerate(getattr(model, "tuning_results", []), start = 1):

            # Log the setting's val F1.
            mlflow.log_metric("tuning/val_f1", result["val_f1"], step = step)

    def log_plots(self, predictions: dict = None):
        """
        Log the confusion matrices, ROC curves and precision-recall curves on train, val and test as images.

        Args:
            predictions (dict): Split name to its 'y', 'proba' and 'pred'.
        """

        # Create one confusion matrix per split, side by side.
        figure, axes = plt.subplots(1, len(predictions), figsize = (5 * len(predictions), 4))

        # Draw each split's confusion matrix.
        for ax, (split, data) in zip(axes, predictions.items()):

            # Draw the matrix with counts.
            ConfusionMatrixDisplay.from_predictions(data["y"], data["pred"], labels = [0, 1], display_labels = ["normal", "extreme"], ax = ax, colorbar = False)

            # Title it with the split.
            ax.set_title(split)

        # Store the figure.
        mlflow.log_figure(figure, "plots/confusion_matrix.png")

        # Free the figure.
        plt.close(figure)

        # Draw the ROC and precision-recall curves of every split.
        for display, file in ((RocCurveDisplay, "roc_curve.png"), (PrecisionRecallDisplay, "precision_recall_curve.png")):

            # Create one chart for all splits.
            figure, ax = plt.subplots(figsize = (6, 5))

            # Add each split's curve.
            for split, data in predictions.items():

                # Draw the curve from the extreme class probabilities.
                display.from_predictions(data["y"], data["proba"], name = split, ax = ax)

            # Store the figure.
            mlflow.log_figure(figure, f"plots/{file}")

            # Free the figure.
            plt.close(figure)

    def register_best(self, best: str = None, uris: dict = None, table: pd.DataFrame = None) -> dict:
        """
        Register the best model as a new version, tag it, and point the environment's alias at it.
        With promote_only_if_better, the alias only moves if the new version's val F1 beats the version it points at.

        Args:
            best (str): Name of the best model.
            uris (dict): Model name to its logged model URI.
            table (pd.DataFrame): One row per model and split with the metrics.

        Returns:
            dict: The registered 'version' and whether the alias was 'promoted'.
        """

        # Create the registry client.
        client = MlflowClient()

        # Get the best model's F1 per split.
        f1 = table[table["model"] == best].set_index("split")["f1"]

        # Register the best model as a new version.
        version = mlflow.register_model(model_uri = uris[best], name = self.registered_model).version

        # Tag the version with what it is and how it scored.
        for key, value in {"environment": self.env, "model": best, "val_f1": f"{f1['val']:.6f}", "test_f1": f"{f1['test']:.6f}"}.items():

            # Set the tag.
            client.set_model_version_tag(self.registered_model, version, key, value)

        # Start by moving the alias.
        promote = True

        # Compare with the current version if only better models may be promoted.
        if self.promote_only_if_better:

            # Read the val F1 of the version the alias points at.
            try:

                # Get the current version's val F1.
                current = float(client.get_model_version_by_alias(self.registered_model, self.env).tags.get("val_f1", "-1"))

            except mlflow.exceptions.MlflowException:

                # No current version yet.
                current = -1.0

            # Move the alias only if the new version is better.
            promote = f1["val"] > current

        # Point the environment's alias at the new version if it should move.
        if promote:

            # Set the alias.
            client.set_registered_model_alias(self.registered_model, self.env, version)

        # Return the outcome.
        return {"version": version, "promoted": promote}
