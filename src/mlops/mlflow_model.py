import joblib
import mlflow.pyfunc
import pandas as pd

class MlflowModel(mlflow.pyfunc.PythonModel):
    """
    MLflow wrapper around a saved TemporalCNNModel, so the registry can store and load it in MLflow's standard format.
    """

    def load_context(self, context):
        """
        Load the saved model from the artifact MLflow stored with it.

        Args:
            context (mlflow.pyfunc.PythonModelContext): MLflow context holding the 'model' artifact path.
        """

        # Load the saved model.
        self.model = joblib.load(context.artifacts["model"])

    def predict(self, context, model_input: pd.DataFrame = None) -> pd.DataFrame:
        """
        Predict on a block of consecutive trading days. The first window - 1 days only serve as history.

        Args:
            context (mlflow.pyfunc.PythonModelContext): MLflow context.
            model_input (pd.DataFrame): Consecutive days with a 'Date' column, the model's features and its target column.

        Returns:
            pd.DataFrame: Date, probability of an extreme day tomorrow, and prediction for each day with a full window.
        """

        # Predict on the block as a single split.
        predictions = self.model.predict(splits = {"data": model_input})["data"]

        # Return one row per predicted day.
        return pd.DataFrame({"Date": predictions["dates"], "proba": predictions["proba"], "pred": predictions["pred"]})
