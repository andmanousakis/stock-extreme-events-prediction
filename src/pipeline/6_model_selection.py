import shutil
from pathlib import Path
import pandas as pd
from src.mlops.mlflow_tracker import MlflowTracker
from utils.config_loader import ConfigLoader
from utils.storage_io import StorageIO
from utils.error_handler import log_step
from utils.logger import Logger

class ModelSelection:
    """
    Pick the best model of the latest pipeline run by its validation score, save it as the best model,
    and register it in MLflow with the environment's alias. Test is only reported, never used to choose.
    """

    def __init__(self):
        """
        Create the logger and load the model selection configs.
        """

        # Create a logger named after the class.
        self.logger = Logger().get_logger(self.__class__.__name__)

        # Load configs.
        self.config = ConfigLoader().load("config-model-selection")

        # Get the input configs.
        self.input = self.config["input"]

        # Get the metric used to pick the best model.
        self.selection_metric = self.config["selection_metric"]

        # Get the output configs.
        self.output = self.config["output"]

        # Create the MLflow tracker from the tracking configs.
        self.tracker = MlflowTracker(**self.config["tracking"])

        # Create the table storage.
        self.storage = StorageIO()

    @log_step("Load metrics")
    def load_metrics(self) -> pd.DataFrame:
        """
        Read the metrics table and keep only the latest pipeline run's rows.

        Returns:
            pd.DataFrame: The latest run's metrics, one row per model and split.
        """

        # Read the metrics table.
        metrics = self.storage.read(table = self.input["metrics_table"], all_runs = True)

        # Find the latest run.
        latest = metrics.sort_values("run_date").iloc[-1]["run_id"]

        # Log which run is used.
        self.logger.info(f"Selecting from run {latest}.")

        # Keep the latest run's rows.
        return metrics[metrics["run_id"] == latest].reset_index(drop = True)

    @log_step("Select best")
    def select_best(self, metrics: pd.DataFrame = None) -> str:
        """
        Pick the model with the best validation score. Test scores are not considered.

        Args:
            metrics (pd.DataFrame): The latest run's metrics.

        Returns:
            str: Name of the best model.
        """

        # Get the validation rows.
        val = metrics[metrics["split"] == "val"]

        # Pick the best validation score.
        best = val.sort_values(self.selection_metric, ascending = False).iloc[0]["model"]

        # Log the choice.
        self.logger.info(f"Best model by val {self.selection_metric}: {best}.")

        # Return the best model's name.
        return best

    @log_step("Save best model")
    def save_best(self, best: str = None) -> str:
        """
        Copy the best model's saved file to the best model path.

        Args:
            best (str): Name of the best model.

        Returns:
            str: The best model path.
        """

        # Create the output directory if it does not exist.
        Path(self.output["best_model"]).parent.mkdir(parents = True, exist_ok = True)

        # Copy the best model.
        shutil.copyfile(Path(self.input["models_dir"]) / f"{best}.joblib", self.output["best_model"])

        # Return the best model path.
        return self.output["best_model"]

    @log_step("MLflow registration")
    def register(self, best: str = None, metrics: pd.DataFrame = None) -> dict:
        """
        Register the best model and move the environment's alias. Skipped when it was not logged to MLflow.

        Args:
            best (str): Name of the best model.
            metrics (pd.DataFrame): The latest run's metrics, with each model's 'model_uri'.

        Returns:
            dict: The registered version and whether the alias moved, empty when skipped.
        """

        # Get the best model's logged model URI.
        model_uri = metrics.loc[metrics["model"] == best, "model_uri"].iloc[0]

        # Skip registration without a tracking server or a logged model.
        if not self.tracker.enabled or not model_uri:

            # Log the skip.
            self.logger.info("No MLflow tracking server or logged model, registration skipped.")

            # Return nothing registered.
            return {}

        # Register the best model.
        result = self.tracker.register_best(best = best, model_uri = model_uri, table = metrics, metric = self.selection_metric)

        # Log the outcome.
        self.logger.info(f"Registered {self.tracker.registered_model} version {result['version']} ({best}); alias '{self.tracker.env}' {'moved to it' if result['promoted'] else 'kept on the better existing version'}.")

        # Return the outcome.
        return result

    def pipeline(self):
        """
        Run the model selection steps in order.
        """

        # Load the latest run's metrics.
        metrics = self.load_metrics()

        # Pick the best model on val.
        best = self.select_best(metrics = metrics)

        # Save it as the best model.
        self.save_best(best = best)

        # Register it in MLflow.
        self.register(best = best, metrics = metrics)

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

    # Run model selection.
    ModelSelection.run()
