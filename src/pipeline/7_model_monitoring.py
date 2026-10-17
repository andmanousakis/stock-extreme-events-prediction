from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from src.mlops.mlflow_tracker import MlflowTracker
from utils.config_loader import ConfigLoader
from utils.storage_io import StorageIO
from utils.error_handler import log_step
from utils.logger import Logger

class ModelMonitoring:
    """
    Compare the deployed model with the latest run's best model on the same splits, and move the environment's alias
    to the new model if it scores better on val. The new model is always deployed when nothing is deployed yet.
    Test is only reported, never used to decide.
    """

    # Splits the models are compared on.
    SPLITS = ("val", "test")

    def __init__(self):
        """
        Create the logger and load the model monitoring configs.
        """

        # Create a logger named after the class.
        self.logger = Logger().get_logger(self.__class__.__name__)

        # Load configs.
        self.config = ConfigLoader().load("config-model-monitoring")

        # Get the input configs.
        self.input = self.config["input"]

        # Get the comparison configs.
        self.comparison = self.config["comparison"]

        # Get the output configs.
        self.output = self.config["output"]

        # Create the MLflow tracker from the tracking configs.
        self.tracker = MlflowTracker(**self.config["tracking"])

        # Create the table storage.
        self.storage = StorageIO()

        # Record when this step ran.
        self.run_date = datetime.now(timezone.utc)

    @log_step("Load latest run")
    def load_run_id(self) -> str:
        """
        Find the latest pipeline run in the metrics table.

        Returns:
            str: The latest run's ID.
        """

        # Read the metrics table.
        metrics = self.storage.read(table = self.input["metrics_table"], all_runs = True)

        # Find the latest run.
        run_id = metrics.sort_values("run_date").iloc[-1]["run_id"]

        # Log which run is used.
        self.logger.info(f"Monitoring run {run_id}.")

        # Return the run ID.
        return run_id

    @log_step("Load splits")
    def load_splits(self) -> dict:
        """
        Load the train, val and test sets stored by the data split. Train is only used as history for the first val windows.

        Returns:
            dict: 'train', 'val' and 'test' DataFrames.
        """

        # Read each split table.
        return {name: self.storage.read(table = self.input[name]) for name in ("train", "val", "test")}

    @log_step("Find versions")
    def find_versions(self, run_id: str = None) -> dict:
        """
        Find the version the run registered and the version currently deployed in this environment.

        Args:
            run_id (str): The latest run's ID.

        Returns:
            dict: 'candidate' and 'deployed' versions, each None if missing.
        """

        # Find both versions.
        versions = {"candidate": self.tracker.run_version(run_id = run_id), "deployed": self.tracker.deployed_version()}

        # Log them.
        self.logger.info(f"Candidate version {versions['candidate']}, deployed version {versions['deployed']}.")

        # Return the versions.
        return versions

    @log_step("Score versions")
    def score(self, versions: dict = None, splits: dict = None) -> pd.DataFrame:
        """
        Load each version and score it on val and test with the same metrics as training and evaluation.
        A deployed version that cannot predict on the current splits, e.g. after a feature change, is left out.

        Args:
            versions (dict): 'candidate' and 'deployed' versions.
            splits (dict): 'train', 'val' and 'test' DataFrames.

        Returns:
            pd.DataFrame: One row per role and split.
        """

        # Start with no rows.
        rows = []

        # Score each version that exists, once even if both roles point at it.
        for role, version in versions.items():

            # Skip a missing version.
            if version is None:

                # Move to the next role.
                continue

            # Predict with the version on every split.
            try:

                # Load the version and predict.
                predictions = self.tracker.load(version = version).predict(splits = splits)

            except KeyError as error:

                # Log why the version was left out.
                self.logger.warning(f"{role} version {version} cannot predict on the current splits ({error}), left out of the comparison.")

                # Move to the next role.
                continue

            # Score each compared split.
            for split in self.SPLITS:

                # Get the labels, probabilities and predictions.
                y, proba, pred = predictions[split]["y"], predictions[split]["proba"], predictions[split]["pred"]

                # Add the split's metrics for the extreme class.
                rows.append({
                    "role": role,
                    "version": version,
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

    @log_step("Decide promotion")
    def decide(self, versions: dict = None, table: pd.DataFrame = None) -> bool:
        """
        Decide whether the candidate replaces the deployed version: always when nothing comparable is deployed,
        otherwise only if the candidate scores higher on the comparison split.

        Args:
            versions (dict): 'candidate' and 'deployed' versions.
            table (pd.DataFrame): One row per role and split.

        Returns:
            bool: Whether to move the alias to the candidate.
        """

        # Get the comparison metric and split.
        metric, split = self.comparison["metric"], self.comparison["split"]

        # Get each role's score on the comparison split.
        scores = table[table["split"] == split].set_index("role")[metric]

        # Promote if nothing comparable is deployed.
        if "deployed" not in scores:

            # Log the reason.
            self.logger.info("No comparable deployed version, promoting the candidate.")

            # Promote.
            return True

        # Promote only if the candidate beats the deployed version.
        promote = bool(scores["candidate"] > scores["deployed"])

        # Log the comparison.
        self.logger.info(f"{split} {metric}: candidate {scores['candidate']:.4f} vs deployed {scores['deployed']:.4f}, {'promoting' if promote else 'keeping the deployed version'}.")

        # Return the decision.
        return promote

    @log_step("Promote")
    def promote(self, versions: dict = None, table: pd.DataFrame = None, promote: bool = None):
        """
        Tag the candidate with the comparison and move the environment's alias to it if promoted.

        Args:
            versions (dict): 'candidate' and 'deployed' versions.
            table (pd.DataFrame): One row per role and split.
            promote (bool): Whether to move the alias.
        """

        # Get the comparison metric and split.
        metric, split = self.comparison["metric"], self.comparison["split"]

        # Get each role's score on the comparison split.
        scores = table[table["split"] == split].set_index("role")[metric]

        # Tag the candidate with what it was compared against and the outcome.
        self.tracker.tag(version = versions["candidate"], tags = {"compared_with": versions["deployed"] or "none", f"deployed_{split}_{metric}": f"{scores['deployed']:.6f}" if "deployed" in scores else "none", "promoted": promote})

        # Move the alias if promoted.
        if promote:

            # Point the environment's alias at the candidate.
            self.tracker.promote(version = versions["candidate"])

        # Log the outcome.
        self.logger.info(f"Alias '{self.tracker.env}' {'moved to version ' + versions['candidate'] if promote else 'kept on version ' + str(versions['deployed'])}.")

    @log_step("Store monitoring table")
    def write_table(self, table: pd.DataFrame = None, run_id: str = None, promote: bool = None) -> pd.DataFrame:
        """
        Store the comparison as this run's file of the monitoring table.

        Args:
            table (pd.DataFrame): One row per role and split.
            run_id (str): The latest run's ID.
            promote (bool): Whether the alias moved to the candidate.

        Returns:
            pd.DataFrame: The stored table.
        """

        # Add the run and the decision to every row.
        table = table.assign(run_id = run_id, run_date = self.run_date, environment = self.tracker.env, promoted = promote)

        # Write the table.
        self.storage.write(df = table, table = self.output["tables"]["monitoring"])

        # Return the table.
        return table

    def pipeline(self):
        """
        Run the model monitoring steps in order.
        """

        # Skip without a tracking server, since the versions live in the registry.
        if not self.tracker.enabled:

            # Log the skip.
            self.logger.info("No MLflow tracking server, model monitoring skipped.")

            # Stop.
            return

        # Find the latest run.
        run_id = self.load_run_id()

        # Find the candidate and deployed versions.
        versions = self.find_versions(run_id = run_id)

        # Stop if the run registered nothing.
        if versions["candidate"] is None:

            # Log the skip.
            self.logger.info(f"Run {run_id} registered no model, model monitoring skipped.")

            # Stop.
            return

        # Load the splits.
        splits = self.load_splits()

        # Score both versions on the same splits.
        table = self.score(versions = versions, splits = splits)

        # Decide whether to promote the candidate.
        promote = self.decide(versions = versions, table = table)

        # Tag the candidate and move the alias if promoted.
        self.promote(versions = versions, table = table, promote = promote)

        # Store the comparison.
        self.write_table(table = table, run_id = run_id, promote = promote)

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

    # Run model monitoring.
    ModelMonitoring.run()
