import os
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from src.mlops.mlflow_tracker import MlflowTracker
from src.models.temporal_cnn import TemporalCNNModel
from utils.config_loader import ConfigLoader
from utils.delta_io import DeltaIO
from utils.error_handler import log_step
from utils.logger import Logger

class ModelTrainingEvaluation:
    """
    Train every configured model, score it on train, val and test, save it, log it to MLflow,
    and append its metrics and predictions to Delta tables tagged with the pipeline run.
    """

    # Model classes the config can refer to by name.
    MODEL_CLASSES = {"TemporalCNNModel": TemporalCNNModel}

    def __init__(self):
        """
        Create the logger, load the training and evaluation configs, and identify the pipeline run.
        """

        # Create a logger named after the class.
        self.logger = Logger().get_logger(self.__class__.__name__)

        # Load configs.
        self.config = ConfigLoader().load("config-model-training-eval")

        # Get the input configs.
        self.input = self.config["input"]

        # Get the target configs.
        self.target = self.config["target"]

        # Get the model configs.
        self.models_config = self.config["models"]

        # Get the output configs.
        self.output = self.config["output"]

        # Create the MLflow tracker from the tracking configs.
        self.tracker = MlflowTracker(**self.config["tracking"])

        # Create the Delta table reader and writer.
        self.delta = DeltaIO()

        # Get the run start time in UTC.
        self.run_date = datetime.now(timezone.utc)

        # Use the CI run ID if given, otherwise a local one from the start time.
        self.run_id = os.environ.get("PIPELINE_RUN_ID") or self.run_date.strftime("local-%Y%m%dT%H%M%S")

    @log_step("Load splits")
    def load_splits(self) -> dict:
        """
        Load the train, val and test sets stored by the data split.

        Returns:
            dict: 'train', 'val' and 'test' DataFrames.
        """

        # Read each split table.
        return {name: self.delta.read(table = self.input[name]) for name in ("train", "val", "test")}

    @log_step("Build models")
    def build_models(self) -> dict:
        """
        Create one model per config entry from its class name and settings.

        Returns:
            dict: Model name to untrained model.
        """

        # Create each model with the shared target column.
        return {name: self.MODEL_CLASSES[entry["class"]](target = self.target["column"], **entry["params"]) for name, entry in self.models_config.items()}

    @log_step("Train models")
    def fit_models(self, models: dict = None, splits: dict = None) -> dict:
        """
        Train every model on train, using val for its own tuning or early stopping.

        Args:
            models (dict): Model name to untrained model.
            splits (dict): 'train', 'val' and 'test' DataFrames.

        Returns:
            dict: Model name to trained model.
        """

        # Train each model.
        for name, model in models.items():

            # Train the model.
            model.fit(splits = splits)

            # Log the training outcome.
            self.logger.info(f"{name}: best epoch {model.best_epoch} of {len(model.history)}, threshold {model.threshold:.2f}.")

        # Return the trained models.
        return models

    @log_step("Predict")
    def predict(self, models: dict = None, splits: dict = None) -> dict:
        """
        Predict with every trained model on every split.

        Args:
            models (dict): Model name to trained model.
            splits (dict): 'train', 'val' and 'test' DataFrames.

        Returns:
            dict: Model name to its predictions per split.
        """

        # Predict with each trained model.
        return {name: model.predict(splits = splits) for name, model in models.items()}

    @log_step("Scoring")
    def score(self, predictions: dict = None) -> pd.DataFrame:
        """
        Score every model on every split with accuracy, precision, recall, F1, ROC AUC and MCC.

        Args:
            predictions (dict): Model name to its predictions per split.

        Returns:
            pd.DataFrame: One row per model and split.
        """

        # Start with no rows.
        rows = []

        # Score each model and split.
        for model, splits in predictions.items():

            # Score each split.
            for split, data in splits.items():

                # Get the labels, probabilities and predictions.
                y, proba, pred = data["y"], data["proba"], data["pred"]

                # Add the split's metrics for the extreme class.
                rows.append({
                    "model": model,
                    "split": split,
                    "samples": len(y),
                    "accuracy": accuracy_score(y, pred),
                    "precision": precision_score(y, pred, zero_division = 0),
                    "recall": recall_score(y, pred, zero_division = 0),
                    "f1": f1_score(y, pred, zero_division = 0),
                    "roc_auc": roc_auc_score(y, proba) if len(set(y)) == 2 else np.nan,
                    "mcc": matthews_corrcoef(y, pred),
                })

        # Return the table.
        return pd.DataFrame(rows)

    @log_step("Report")
    def report(self, table: pd.DataFrame = None) -> pd.DataFrame:
        """
        Log each model's test metrics and its F1 across splits to check for overfitting.

        Args:
            table (pd.DataFrame): One row per model and split.

        Returns:
            pd.DataFrame: The same table.
        """

        # Log the test-set metrics, best F1 first.
        for row in table[table["split"] == "test"].sort_values("f1", ascending = False).itertuples():

            # Log the model's test metrics.
            self.logger.info(f"test  {row.model:<15} F1 {row.f1:.3f}, precision {row.precision:.3f}, recall {row.recall:.3f}, MCC {row.mcc:.3f}, ROC AUC {row.roc_auc:.3f}, accuracy {row.accuracy:.3f}.")

        # Log each model's F1 across splits.
        for name in self.models_config:

            # Get the model's F1 per split.
            f1 = table[table["model"] == name].set_index("split")["f1"]

            # Log the overfitting check.
            self.logger.info(f"F1 by split  {name:<15} train {f1['train']:.3f}, val {f1['val']:.3f}, test {f1['test']:.3f}.")

        # Return the table.
        return table

    @log_step("Save models")
    def save_models(self, models: dict = None) -> dict:
        """
        Save every trained model to <models_dir>/<name>.joblib.

        Args:
            models (dict): Model name to trained model.

        Returns:
            dict: The saved models.
        """

        # Create the models directory if it does not exist.
        Path(self.output["models_dir"]).mkdir(parents = True, exist_ok = True)

        # Save each trained model.
        for name, model in models.items():

            # Save the model.
            joblib.dump(model, Path(self.output["models_dir"]) / f"{name}.joblib")

        # Return the models.
        return models

    @log_step("MLflow tracking")
    def track(self, models: dict = None, predictions: dict = None, table: pd.DataFrame = None) -> dict:
        """
        Log every model to MLflow as its own run. Skipped when no tracking server is configured.

        Args:
            models (dict): Model name to trained model.
            predictions (dict): Model name to its predictions per split.
            table (pd.DataFrame): One row per model and split.

        Returns:
            dict: Model name to its 'mlflow_run_id' and 'model_uri', empty when tracking is off.
        """

        # Skip tracking without a tracking server.
        if not self.tracker.enabled:

            # Log the skip.
            self.logger.info("MLFLOW_TRACKING_URI not set, MLflow tracking off.")

            # Return no runs.
            return {}

        # Log the models.
        runs = self.tracker.log_models(models_config = self.models_config, models = models, predictions = predictions, table = table, models_dir = self.output["models_dir"])

        # Log how many runs were logged.
        self.logger.info(f"Logged {len(runs)} runs to experiment '{self.tracker.experiment}'.")

        # Return the runs.
        return runs

    def run_columns(self, model: str = None, runs: dict = None) -> dict:
        """
        Build the columns that tie a table row to this pipeline run and the model's MLflow run.

        Args:
            model (str): Model name.
            runs (dict): Model name to its 'mlflow_run_id' and 'model_uri'; empty when tracking is off.

        Returns:
            dict: 'run_id', 'run_date', 'mlflow_run_id' and 'model_uri'.
        """

        # Get the model's MLflow run, if it was tracked.
        run = runs.get(model, {})

        # Return the run columns.
        return {"run_id": self.run_id, "run_date": self.run_date, "mlflow_run_id": run.get("mlflow_run_id", ""), "model_uri": run.get("model_uri", "")}

    @log_step("Write tables")
    def write_tables(self, table: pd.DataFrame = None, predictions: dict = None, runs: dict = None) -> dict:
        """
        Append this run's metrics and predictions to their Delta tables, tagged with the run ID, run date,
        and each model's MLflow run ID and model URI.

        Args:
            table (pd.DataFrame): One row per model and split.
            predictions (dict): Model name to its predictions per split.
            runs (dict): Model name to its 'mlflow_run_id' and 'model_uri'.

        Returns:
            dict: Table name to its new version.
        """

        # Add the run columns to the metrics.
        metrics = pd.concat([table[table["model"] == model].assign(**self.run_columns(model = model, runs = runs)) for model in table["model"].unique()], ignore_index = True)

        # Build one prediction row per model, split and day, with the run columns.
        rows = pd.concat([pd.DataFrame({"model": model, "split": split, "date": data["dates"], "y": data["y"], "proba": data["proba"], "pred": data["pred"]}).assign(**self.run_columns(model = model, runs = runs)) for model, splits in predictions.items() for split, data in splits.items()], ignore_index = True)

        # Append both tables.
        versions = {name: self.delta.write(df = df, table = self.output["tables"][name], mode = "append") for name, df in (("metrics", metrics), ("predictions", rows))}

        # Log where they went.
        self.logger.info(f"Run {self.run_id}: metrics table version {versions['metrics']}, predictions table version {versions['predictions']}.")

        # Return the new versions.
        return versions

    def pipeline(self):
        """
        Run the training and evaluation steps in order.
        """

        # Load the splits.
        splits = self.load_splits()

        # Create the configured models.
        models = self.build_models()

        # Train them.
        models = self.fit_models(models = models, splits = splits)

        # Predict with the models.
        predictions = self.predict(models = models, splits = splits)

        # Score everything.
        table = self.score(predictions = predictions)

        # Log the results.
        table = self.report(table = table)

        # Save the models.
        self.save_models(models = models)

        # Log every model to MLflow.
        runs = self.track(models = models, predictions = predictions, table = table)

        # Append the metrics and predictions tables.
        self.write_tables(table = table, predictions = predictions, runs = runs)

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

    # Run model training and evaluation.
    ModelTrainingEvaluation.run()
