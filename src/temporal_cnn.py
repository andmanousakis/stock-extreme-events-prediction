from rich import print
import inspect
import pandas as pd
import numpy as np
from pprint import pprint
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score

class TemporalCNN(nn.Module):

    def __init__(self, num_features: int = 6, sequence_length: int = 10):
        super(TemporalCNN, self).__init__()

        # 1D Convolutional Layers.
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        # Flatten Layer.
        self.flatten = nn.Flatten()
        
        # Fully Connected Layer.
        self.fc = nn.Linear(64 * sequence_length, 2)
        
        # Softmax Layer for Classification.
        self.softmax = nn.Softmax(dim=1)
    
    # region Utilties.
    def class_name(self):

        # Return class name.
        return self.__class__.__name__
    
    def method_name(self):

        # Return caller's method name.
        return inspect.currentframe().f_back.f_code.co_name
    # endregion.

    # region Data Preparation.
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

    def input_preparation(self, training_set: np.ndarray = None):

        try:

            # Define sequence length (10 days).
            sequence_length = 10

            # Prepare sequences and corresponding targets.
            X = []
            y = []

            # Create sequences.
            for i in range(len(training_set['features']) - sequence_length):

                # Extract a sequence of 10 days.
                X.append(training_set['features'][i:i + sequence_length])

                # Target is the label for the next day.
                y.append(training_set['target'][i + sequence_length])
            
            # Convert to NumPy arrays.
            X = np.array(X)
            y = np.array(y)

            # Reshape X_train to [batch_size, num_features, sequence_length] for TCNN.
            X = np.transpose(X, (0, 2, 1))

            # Print message.
            print(f"[green]Success[/green] -> {self.class_name()} > {self.method_name()}: Prepping data for Temporal CNN.")

            # Display shapes to confirm.
            print(f"Shape of X_train: {X.shape}")  # Expected: (n_samples, num_features, sequence_length)
            print(f"Shape of y_train: {y.shape}")  # Expected: (n_samples,)
        
        except Exception as e:

            # Print message.
            print(f"[red]Error[/red] -> {self.class_name()} > {self.method_name()}: {e}.")

            # Exit.
            raise SystemExit
        
        return {
            "X": X,
            "Y": y
        }
    # endregion.

    # region Deep Learning pipeline.
    def forward(self, x):
        # Pass through first Conv layer + ReLU
        x = self.relu1(self.conv1(x))
        
        # Pass through second Conv layer + ReLU
        x = self.relu2(self.conv2(x))
        
        # Flatten the tensor for Dense layer
        x = self.flatten(x)
        
        # Pass through Dense layer
        x = self.fc(x)
        
        # Softmax for class probabilities
        # x = self.softmax(x)
        
        return x

    def train_validate_model(self, model, training_set: np.ndarray = None, validation_set: np.ndarray = None, num_epochs = 50, learning_rate = 0.001, bs=10, patience=5):

        try:

            # Convert data to PyTorch tensors and create data loaders.
            train_loader = DataLoader(
                TensorDataset(torch.tensor(training_set['X'], dtype=torch.float32), torch.tensor(training_set['Y'], dtype=torch.long).squeeze()),
                batch_size=bs, shuffle=True
            )
            val_loader = DataLoader(
                TensorDataset(torch.tensor(validation_set['X'], dtype=torch.float32), torch.tensor(validation_set['Y'], dtype=torch.long).squeeze()),
                batch_size=bs, shuffle=False
            )

            # Define Loss and Optimizer.
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            # Early stopping variables.
            best_val_loss = float('inf')
            patience_counter = 0

            # Training loop.
            for epoch in range(num_epochs):
                model.train()
                train_loss = 0.0

                for inputs, labels in train_loader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels.view(-1))
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                # Validation.
                model.eval()
                correct, total = 0, 0
                all_labels = []
                all_predictions = []
                val_loss = 0.0
                
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels.view(-1))
                        val_loss += loss.item()

                        _, predicted = torch.max(outputs, 1)
                        total += labels.size(0)
                        correct += (predicted == labels.view(-1)).sum().item()

                        # Store all labels and predictions for metrics.
                        all_labels.extend(labels.cpu().numpy())
                        all_predictions.extend(predicted.cpu().numpy())

                val_accuracy = 100 * correct / total
                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
                print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

                # Early Stopping Check.
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0  # Reset counter if validation loss improves
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping triggered after {epoch+1} epochs.")
                        break

            # Compute and print evaluation metrics.
            cm = confusion_matrix(all_labels, all_predictions)
            precision = precision_score(all_labels, all_predictions, average='binary')
            recall = recall_score(all_labels, all_predictions, average='binary')
            f1 = f1_score(all_labels, all_predictions, average='binary')

            print("\nConfusion Matrix:")
            print(cm)
            print("\nClassification Report:")
            print(classification_report(all_labels, all_predictions, digits=4))
            print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

            # Print message.
            print(f"[green]Success[/green] -> {self.class_name()} > {self.method_name()}: Model training & validation.")

        except Exception as e:
            
            # Print message.
            print(f"[red]Error[/red] -> {self.class_name()} > {self.method_name()}: {e}")

            # Exit.
            raise SystemExit
        
        # Return model.
        return model
    
    def test_model(self, model, test_set: np.ndarray = None, bs=64):

        try:

            # Convert test data to PyTorch tensors and create a data loader.
            test_loader = DataLoader(
                TensorDataset(torch.tensor(test_set['X'], dtype=torch.float32), torch.tensor(test_set['Y'], dtype=torch.long).squeeze()),
                batch_size=bs, shuffle=False
            )

            # Set model to evaluation mode.
            model.eval()
            correct, total = 0, 0
            all_labels = []
            all_predictions = []

            # Disable gradient computation for evaluation.
            with torch.no_grad():
                for inputs, labels in test_loader:
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels.view(-1)).sum().item()

                    # Store all labels and predictions for metrics calculation.
                    all_labels.extend(labels.cpu().numpy())
                    all_predictions.extend(predicted.cpu().numpy())

            # Compute evaluation metrics.
            accuracy = 100 * correct / total
            precision = precision_score(all_labels, all_predictions, average='binary')
            recall = recall_score(all_labels, all_predictions, average='binary')
            f1 = f1_score(all_labels, all_predictions, average='binary')
            cm = confusion_matrix(all_labels, all_predictions)

            # Print results.
            print(f"Test Accuracy: {accuracy:.2f}%")
            print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
            print("\nConfusion Matrix:")
            print(cm)
            print("\nClassification Report:")
            print(classification_report(all_labels, all_predictions, digits=4))

            # Print message.
            print(f"[green]Success[/green] -> {self.class_name()} > {self.method_name()}: Model evaluation.")

        except Exception as e:
            
            # Print message.
            print(f"[red]Error[/red] -> {self.class_name()} > {self.method_name()}: {e}")

            # Exit.
            raise SystemExit
    
    def workflow(self):

        # Read training, validation, testing sets.
        datasets_dict= self.read_data()

        # Step 3.1: Prep data for temporal predictive modeling.
        training_set = self.input_preparation(datasets_dict['training_set'])
        validation_set = self.input_preparation(datasets_dict['validation_set'])
        test_set = self.input_preparation(datasets_dict['test_set'])

        # Step 3.2: Model architecture > It has already been instantiated in the constructor method __init__.
        tcnn_model = TemporalCNN(num_features = 6, sequence_length = 10)

        # Step 3.3: Model training.
        tcnn_model = self.train_validate_model(model = tcnn_model, training_set = training_set, validation_set = validation_set, num_epochs = 200, learning_rate = 0.001, bs = 10, patience = 15)

        # Step 3.4: Model Evaluation.
        self.test_model(model = tcnn_model, test_set = test_set, bs = 10)
    # endregion.

    @classmethod
    def run(cls):

        # Instantiate class object.
        instance = cls()

        # Run workflow.
        instance.workflow()
    
if __name__ == "__main__":

    TemporalCNN.run()