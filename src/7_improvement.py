import importlib
from pathlib import Path
import pandas as pd
from utils.config_loader import ConfigLoader
from utils.error_handler import log_step
from utils.logger import Logger

temporal_cnn = importlib.import_module("src.6_temporal_cnn")

class Improvement:
    """
    Run temporal CNN experiments, each changing the baseline settings, compare them on the validation set and save the best.
    """

    def __init__(self):
        """
        Create the logger and load the improvement configs.
        """

        # Create a logger named after the class.
        self.logger = Logger().get_logger(self.__class__.__name__)

        # Load configs.
        self.config = ConfigLoader().load("config-improvement")

        # Get the experiments.
        self.experiments = self.config["experiments"]

        # Get the output configs.
        self.output = self.config["output"]

    @log_step("Load splits")
    def load_splits(self) -> dict:
        """
        Load the train, val and test sets once for all experiments, using the temporal CNN's input paths.

        Returns:
            dict: 'train', 'val' and 'test' DataFrames.
        """

        # Load the splits through the temporal CNN.
        return temporal_cnn.TemporalCNN().load_splits()

    def run_experiment(self, experiment: dict = None, splits: dict = None) -> dict:
        """
        Train the temporal CNN with the baseline settings changed by one experiment, and score it on train and val.

        Args:
            experiment (dict): 'name' and optional 'features', 'network' and 'training' overrides.
            splits (dict): 'train', 'val' and 'test' DataFrames.

        Returns:
            dict: The experiment 'name', its temporal CNN object 'cnn' and its trained 'result'.
        """

        # Create a temporal CNN with the baseline settings.
        cnn = temporal_cnn.TemporalCNN()

        # Use the experiment's features if given.
        cnn.features = experiment.get("features", cnn.features)

        # Override the network settings.
        cnn.network_config.update(experiment.get("network", {}))

        # Override the training settings.
        cnn.training.update(experiment.get("training", {}))

        # Scale features using the training set.
        scaled = cnn.scale_features(splits = splits)

        # Build 10-day sequences.
        windows = cnn.build_windows(splits = scaled)

        # Train with early stopping.
        result = cnn.train_model(network = cnn.build_network(), windows = windows)

        # Pick the decision threshold.
        result = cnn.tune_threshold(result = result, windows = windows)

        # Score on train and val.
        result = cnn.score_model(result = result, windows = windows)

        # Return the experiment.
        return {"name": experiment["name"], "cnn": cnn, "result": result}

    @log_step("Experiments")
    def run_experiments(self, splits: dict = None) -> list:
        """
        Run every configured experiment in order.

        Args:
            splits (dict): 'train', 'val' and 'test' DataFrames.

        Returns:
            list: One finished experiment per configured experiment.
        """

        # Run each experiment.
        return [self.run_experiment(experiment = experiment, splits = splits) for experiment in self.experiments]

    @log_step("Compare experiments")
    def compare(self, runs: list = None) -> pd.DataFrame:
        """
        Build and log a table of each experiment's validation metrics, threshold and best epoch.

        Args:
            runs (list): Finished experiments.

        Returns:
            pd.DataFrame: One row per experiment.
        """

        # Collect one row per experiment.
        table = pd.DataFrame([{
            "experiment": run["name"],
            "val_f1": run["result"]["metrics"]["val"]["f1"],
            "val_precision": run["result"]["metrics"]["val"]["precision"],
            "val_recall": run["result"]["metrics"]["val"]["recall"],
            "train_f1": run["result"]["metrics"]["train"]["f1"],
            "threshold": run["result"]["threshold"],
            "best_epoch": run["result"]["best_epoch"],
        } for run in runs])

        # Log each row.
        for row in table.itertuples():

            # Log the experiment's results.
            self.logger.info(f"{row.experiment:<20} val F1 {row.val_f1:.3f}, precision {row.val_precision:.3f}, recall {row.val_recall:.3f}, train F1 {row.train_f1:.3f}, threshold {row.threshold:.2f}, best epoch {row.best_epoch}.")

        # Return the table.
        return table

    @log_step("Select best")
    def select_best(self, runs: list = None) -> dict:
        """
        Pick the experiment with the highest validation F1.

        Args:
            runs (list): Finished experiments.

        Returns:
            dict: The best experiment.
        """

        # Pick the highest validation F1.
        best = max(runs, key = lambda run: run["result"]["metrics"]["val"]["f1"])

        # Log the choice.
        self.logger.info(f"Best experiment: {best['name']}, val F1 {best['result']['metrics']['val']['f1']:.3f}.")

        # Return the best experiment.
        return best

    @log_step("Save results")
    def save_results(self, table: pd.DataFrame = None, best: dict = None) -> dict:
        """
        Save the comparison table and the best experiment's model.

        Args:
            table (pd.DataFrame): One row per experiment.
            best (dict): The best experiment.

        Returns:
            dict: The best experiment.
        """

        # Create the results directory if it does not exist.
        Path(self.output["results"]).parent.mkdir(parents = True, exist_ok = True)

        # Store the comparison table.
        table.to_csv(self.output["results"], index = False)

        # Save the best model with its own settings.
        best["cnn"].save_model(result = best["result"], path = self.output["model"])

        # Return the best experiment.
        return best

    def pipeline(self):
        """
        Run the improvement steps in order.
        """

        # Load the splits once.
        splits = self.load_splits()

        # Run every experiment.
        runs = self.run_experiments(splits = splits)

        # Compare the experiments.
        table = self.compare(runs = runs)

        # Pick the best experiment.
        best = self.select_best(runs = runs)

        # Save the table and the best model.
        self.save_results(table = table, best = best)

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

    # Run the improvement experiments.
    Improvement.run()
