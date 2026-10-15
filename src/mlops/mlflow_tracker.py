import json
import os
from pathlib import Path
import mlflow
import pandas as pd
from mlflow import MlflowClient
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

    def log_models(self, models_config: dict = None, table: pd.DataFrame = None, models_dir: str = None) -> dict:
        """
        Log one run per model with its settings, its metrics on every split and the saved model in MLflow's format.

        Args:
            models_config (dict): Model name to its config entry ('class' and 'params').
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

                # Log the saved model in MLflow's standard format. Loading it needs this project's code installed, as in its images.
                info = mlflow.pyfunc.log_model(name = "model", python_model = MlflowModel(), artifacts = {"model": str(Path(models_dir) / f"{name}.joblib")})

                # Store the logged model URI.
                uris[name] = info.model_uri

        # Return the logged model URIs.
        return uris

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
