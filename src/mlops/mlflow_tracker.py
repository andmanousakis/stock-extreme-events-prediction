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
    Log model runs to MLflow, register models, load registered versions and move the environment's alias.
    Only active when MLFLOW_TRACKING_URI is set.
    """

    # Metrics logged for every split.
    METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "mcc")

    def __init__(self, experiment: str = None, registered_model: str = None):
        """
        Store the tracking settings and read the environment.

        Args:
            experiment (str): MLflow experiment name.
            registered_model (str): Registered model name in the Model Registry.
        """

        # Store the experiment name.
        self.experiment = experiment

        # Store the registered model name.
        self.registered_model = registered_model

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
            dict: Model name to its 'mlflow_run_id' and logged 'model_uri'.
        """

        # Use the configured experiment.
        mlflow.set_experiment(self.experiment)

        # Start with no logged models.
        runs = {}

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

                # Store the run ID and logged model URI.
                runs[name] = {"mlflow_run_id": mlflow.active_run().info.run_id, "model_uri": info.model_uri}

        # Return the runs.
        return runs

    def log_curves(self, model = None):
        """
        Log the model's training curve by epoch.

        Args:
            model (TemporalCNNModel): Trained model with a per-epoch 'history'.
        """

        # Log each point of the training curve.
        for point in model.history:

            # Log the epoch's losses and F1.
            mlflow.log_metrics({f"training/{key}": value for key, value in point.items() if key != "epoch"}, step = point["epoch"])

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

    def register(self, best: str = None, model_uri: str = None, table: pd.DataFrame = None, metric: str = "mcc") -> str:
        """
        Register the best model of the run as a new version and tag it with what it is and how it scored.
        The environment's alias is not moved here; model monitoring decides that.

        Args:
            best (str): Name of the best model.
            model_uri (str): The best model's logged model URI.
            table (pd.DataFrame): One row per model and split with the metrics.
            metric (str): Metric the model was selected by, e.g. 'mcc'.

        Returns:
            str: The registered version.
        """

        # Get the best model's score per split.
        score = table[table["model"] == best].set_index("split")[metric]

        # Register the best model as a new version.
        version = mlflow.register_model(model_uri = model_uri, name = self.registered_model).version

        # Tag the version with what it is and how it scored.
        self.tag(version = version, tags = {"environment": self.env, "run_id": table["run_id"].iloc[0], "model": best, "metric": metric, f"val_{metric}": f"{score['val']:.6f}", f"test_{metric}": f"{score['test']:.6f}"})

        # Return the version.
        return version

    def run_version(self, run_id: str = None) -> str:
        """
        Get the version registered by a pipeline run in this environment.

        Args:
            run_id (str): Pipeline run ID the version was tagged with.

        Returns:
            str: The run's version, or None if the run registered none.
        """

        # Find the versions registered by the run in this environment.
        versions = MlflowClient().search_model_versions(f"name = '{self.registered_model}' and tag.environment = '{self.env}' and tag.run_id = '{run_id}'")

        # Return the highest version number, if any.
        return str(max(int(v.version) for v in versions)) if versions else None

    def deployed_version(self) -> str:
        """
        Get the version the environment's alias points at.

        Returns:
            str: The deployed version, or None if the alias is not set yet.
        """

        # Read the version behind the alias.
        try:

            # Return its version.
            return MlflowClient().get_model_version_by_alias(self.registered_model, self.env).version

        except mlflow.exceptions.MlflowException:

            # No version deployed yet.
            return None

    def load(self, version: str = None):
        """
        Load a registered version and unwrap it to the project's model class, so it can predict on the pipeline splits.

        Args:
            version (str): Registered version to load.

        Returns:
            TemporalCNNModel: The trained model.
        """

        # Load the version and return the model inside the MLflow wrapper.
        return mlflow.pyfunc.load_model(f"models:/{self.registered_model}/{version}").unwrap_python_model().model

    def tag(self, version: str = None, tags: dict = None):
        """
        Set tags on a registered version.

        Args:
            version (str): Registered version.
            tags (dict): Tag names and values.
        """

        # Create the registry client.
        client = MlflowClient()

        # Set each tag.
        for key, value in tags.items():

            # Set the tag.
            client.set_model_version_tag(self.registered_model, version, key, str(value))

    def promote(self, version: str = None):
        """
        Point the environment's alias at a version.

        Args:
            version (str): Registered version to deploy.
        """

        # Set the alias.
        MlflowClient().set_registered_model_alias(self.registered_model, self.env, version)
