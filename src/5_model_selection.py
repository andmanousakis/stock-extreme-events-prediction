from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from src.mlops.mlflow_tracker import MlflowTracker
from src.models.random_forest import RandomForestModel
from src.models.temporal_cnn import TemporalCNNModel
from utils.config_loader import ConfigLoader
from utils.error_handler import log_step
from utils.logger import Logger

class ModelSelection:
    """
    Train every configured model, score it on train, val and test,
    and keep the best by validation score. Test is only reported, never used to choose.
    When an MLflow server is configured, every model is logged as a run and the best is registered.
    """

    # Model classes the config can refer to by name.
    MODEL_CLASSES = {"RandomForestModel": RandomForestModel, "TemporalCNNModel": TemporalCNNModel}

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

        # Get the target configs.
        self.target = self.config["target"]

        # Get the metric used to pick the best model.
        self.selection_metric = self.config["selection_metric"]

        # Get the model configs.
        self.models_config = self.config["models"]

        # Get the output configs.
        self.output = self.config["output"]

        # Create the MLflow tracker from the tracking configs.
        self.tracker = MlflowTracker(**self.config["tracking"])

    @log_step("Load splits")
    def load_splits(self) -> dict:
        """
        Load the train, val and test sets stored by the data split.

        Returns:
            dict: 'train', 'val' and 'test' DataFrames.
        """

        # Load each split.
        return {name: pd.read_excel(self.input[name]) for name in ("train", "val", "test")}

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

            # Describe what training chose: the forest's settings or the network's best epoch.
            chosen = f"params {model.best_params}" if isinstance(model, RandomForestModel) else f"best epoch {model.best_epoch} of {len(model.history)}"

            # Log the training outcome.
            self.logger.info(f"{name}: {chosen}, threshold {model.threshold:.2f}.")

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

    @log_step("Select best")
    def select_best(self, table: pd.DataFrame = None, models: dict = None) -> str:
        """
        Pick the model with the best validation score. Test scores are not considered.

        Args:
            table (pd.DataFrame): One row per model and split.
            models (dict): Model name to trained model.

        Returns:
            str: Name of the best model.
        """

        # Get the trained models' validation rows.
        val = table[(table["split"] == "val") & (table["model"].isin(models))]

        # Pick the best validation score.
        best = val.sort_values(self.selection_metric, ascending = False).iloc[0]["model"]

        # Log the choice.
        self.logger.info(f"Best model by val {self.selection_metric}: {best}.")

        # Return the best model's name.
        return best

    @log_step("Report")
    def report(self, table: pd.DataFrame = None, models: dict = None) -> pd.DataFrame:
        """
        Log the test-set comparison, best first, and each trained model's F1 across splits to check for overfitting.

        Args:
            table (pd.DataFrame): One row per model and split.
            models (dict): Model name to trained model.

        Returns:
            pd.DataFrame: The same table.
        """

        # Log the test-set comparison, best F1 first.
        for row in table[table["split"] == "test"].sort_values("f1", ascending = False).itertuples():

            # Log the model's test metrics.
            self.logger.info(f"test  {row.model:<15} F1 {row.f1:.3f}, precision {row.precision:.3f}, recall {row.recall:.3f}, MCC {row.mcc:.3f}, ROC AUC {row.roc_auc:.3f}, accuracy {row.accuracy:.3f}.")

        # Log each trained model's F1 across splits.
        for name in models:

            # Get the model's F1 per split.
            f1 = table[table["model"] == name].set_index("split")["f1"]

            # Log the overfitting check.
            self.logger.info(f"F1 by split  {name:<15} train {f1['train']:.3f}, val {f1['val']:.3f}, test {f1['test']:.3f}.")

        # Return the table.
        return table

    @log_step("Save results")
    def save(self, models: dict = None, best: str = None, table: pd.DataFrame = None) -> pd.DataFrame:
        """
        Save every trained model, a copy of the best one, and the evaluation table.

        Args:
            models (dict): Model name to trained model.
            best (str): Name of the best model.
            table (pd.DataFrame): One row per model and split.

        Returns:
            pd.DataFrame: The saved table.
        """

        # Create the models directory if it does not exist.
        Path(self.output["models_dir"]).mkdir(parents = True, exist_ok = True)

        # Save each trained model.
        for name, model in models.items():

            # Save the model.
            joblib.dump(model, Path(self.output["models_dir"]) / f"{name}.joblib")

        # Save the best model.
        joblib.dump(models[best], self.output["best_model"])

        # Create the report directory if it does not exist.
        Path(self.output["report"]).parent.mkdir(parents = True, exist_ok = True)

        # Store the evaluation table.
        table.to_csv(self.output["report"], index = False)

        # Return the table.
        return table

    @log_step("MLflow tracking")
    def track(self, models: dict = None, predictions: dict = None, table: pd.DataFrame = None) -> dict:
        """
        Log every model to MLflow as its own run. Skipped when no tracking server is configured.

        Args:
            models (dict): Model name to trained model.
            predictions (dict): Model name to its predictions per split.
            table (pd.DataFrame): One row per model and split.

        Returns:
            dict: Model name to its logged model URI, empty when tracking is off.
        """

        # Skip tracking without a tracking server.
        if not self.tracker.enabled:

            # Log the skip.
            self.logger.info("MLFLOW_TRACKING_URI not set, MLflow tracking off.")

            # Return no runs.
            return {}

        # Log the models.
        uris = self.tracker.log_models(models_config = self.models_config, models = models, predictions = predictions, table = table, models_dir = self.output["models_dir"])

        # Log how many runs were logged.
        self.logger.info(f"Logged {len(uris)} runs to experiment '{self.tracker.experiment}'.")

        # Return the logged model URIs.
        return uris

    @log_step("MLflow registration")
    def register(self, best: str = None, uris: dict = None, table: pd.DataFrame = None) -> dict:
        """
        Register the best model and move the environment's alias. Skipped when tracking is off.

        Args:
            best (str): Name of the best model.
            uris (dict): Model name to its logged model URI.
            table (pd.DataFrame): One row per model and split.

        Returns:
            dict: The registered version and whether the alias moved, empty when tracking is off.
        """

        # Skip registration without tracking.
        if not uris:

            # Return nothing registered.
            return {}

        # Register the best model.
        result = self.tracker.register_best(best = best, uris = uris, table = table)

        # Log the outcome.
        self.logger.info(f"Registered {self.tracker.registered_model} version {result['version']} ({best}); alias '{self.tracker.env}' {'moved to it' if result['promoted'] else 'kept on the better existing version'}.")

        # Return the outcome.
        return result

    def pipeline(self):
        """
        Run the model selection steps in order.
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

        # Pick the best model on val.
        best = self.select_best(table = table, models = models)

        # Log the comparison.
        table = self.report(table = table, models = models)

        # Save the models and report.
        self.save(models = models, best = best, table = table)

        # Log every model to MLflow.
        uris = self.track(models = models, predictions = predictions, table = table)

        # Register the best model.
        self.register(best = best, uris = uris, table = table)

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
