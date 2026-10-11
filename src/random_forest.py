from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import pandas as pd
from rich import print
import inspect
import numpy as np


class RandomForest:

    def __init__(self):
        pass

    def class_name(self):

        # Return class name.
        return self.__class__.__name__
    
    def method_name(self):

        return inspect.currentframe().f_back.f_code.co_name

    def read_data(self):
        
        try:

            # Read training set.
            training_set_df = pd.read_excel("training_set.xlsx")
            X_train = training_set_df[['Open', 'High', 'Low', 'Close', 'Volume', 'daily_return']].to_numpy()
            y_train = training_set_df[['extreme_event']].to_numpy()

            # Read validation set.
            validation_set_df = pd.read_excel("validation_set.xlsx")
            X_val = validation_set_df[['Open', 'High', 'Low', 'Close', 'Volume', 'daily_return']].to_numpy()
            y_val = validation_set_df[['extreme_event']].to_numpy()

            # Read test set.
            test_set_df = pd.read_excel("test_set.xlsx")
            X_test = test_set_df[['Open', 'High', 'Low', 'Close', 'Volume', 'daily_return']].to_numpy()
            y_test = test_set_df[['extreme_event']].to_numpy()

            # Print message.
            print(f"[green]Success[/green] -> {self.class_name()} > {self.method_name()}: Reading training, validation, test sets.")
        
        except Exception as e:

            # Print message.
            print(f"[red]Error[/red] -> {self.class_name()} > {self.method_name()}: {e}.")

            # Exit.
            raise SystemExit
        
        # Return data.
        return {
            "training_set": {
                "features": X_train,
                "target": y_train
            },
            "validation_set": {
                "features": X_val,
                "target": y_val
            },
            "test_set": {
                "features": X_test,
                "target": y_test
            }
        }

    def training(self, training_set: np.ndarray = None):
        
        try:

            # Get the last 10 days for training.
            X_train_last_10 = training_set['features'][-10:]
            y_train_last_10 = training_set['target'][-10:].ravel() # -> convert to 1D array with .ravel.

            # Initialize Random Forest with default hyperparameters.
            rf_model = RandomForestClassifier(n_estimators=500, random_state = 42)
            rf_model.fit(X_train_last_10, y_train_last_10)
            
            # Print message.
            print(f"[green]Success[/green] -> {self.class_name()} > {self.method_name()}: Model training.")

        except Exception as e:

            # Print message.
            print(f"[red]Error[/red] -> {self.class_name()} > {self.method_name()}: {e}")

            # Exit.
            raise SystemExit
        
        # Return model.
        return rf_model

    def validation(self, rf_model, validation_set: np.ndarray = None):

        try:

            # Prep validation set.
            X_val = validation_set['features']
            y_val = validation_set['target'].ravel() # -> convert to 1D array with .ravel.

            # Predict on validation set for hyperparameter tuning if needed.
            val_predictions = rf_model.predict(X_val)

            # Print message.
            print(f"[green]Success[/green] -> {self.class_name()} > {self.method_name()}: Model validation.")

            # Precision, Recall, F1-Score.
            print("\nClassification Report:")
            print(classification_report(y_val, val_predictions))

            # Confusion Matrix.
            conf_matrix = confusion_matrix(y_val, val_predictions)
            print("\nConfusion Matrix:\n", conf_matrix)

        except Exception as e:
            
            # Print message.
            print(f"[red]Error[/red] -> {self.class_name()} > {self.method_name()}: {e}.")

            # Exit.
            raise SystemExit

    def testing(self, rf_model, test_set: np.ndarray = None):

        try:

            # Prep test set.
            X_test = test_set['features']
            y_test = test_set['target'].ravel() # -> convert to 1D array with .ravel.

            # Predict on the test set
            test_predictions = rf_model.predict(X_test)

            # Print message.
            print(f"[green]Success[/green] -> {self.class_name()} > {self.method_name()}: Model evaluation.")

            # Precision, Recall, F1-Score.
            print("\nClassification Report:")
            print(classification_report(y_test, test_predictions))

            # Confusion Matrix.
            conf_matrix = confusion_matrix(y_test, test_predictions)
            print("\nConfusion Matrix:\n", conf_matrix)

        except Exception as e:

            # Print message.
            print(f"[red]Error[/red] -> {self.class_name()} > {self.method_name()}: {e}.")

            # Exit.
            raise SystemExit

    def workflow(self):
        
        # Read training, validation, testing sets.
        datasets_dict= self.read_data()

        # Step 2.1: Model training & validation.
        rf_model = self.training(training_set = datasets_dict['training_set'])
        self.validation(rf_model = rf_model, validation_set = datasets_dict['validation_set'])

        # Step 2.2: Model evaluation.
        self.testing(rf_model = rf_model, test_set = datasets_dict['test_set'])

    @classmethod
    def run(cls):

        # Instantiate class object.
        instance = cls()

        # Run the workflow.
        instance.workflow()


if __name__ == "__main__":

    # Run random forest classifier.
    RandomForest.run()