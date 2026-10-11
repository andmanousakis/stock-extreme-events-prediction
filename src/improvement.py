from rich import print
import sys
import inspect
import pandas as pd
import numpy as np
from pprint import pprint
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import numpy as np
import pandas as pd
import inspect
import matplotlib.pyplot as plt

class LSTMAutoencoder(nn.Module):

    def __init__(self, input_dim=6, hidden_dim=256, num_layers=4, seq_length=10):
        super(LSTMAutoencoder, self).__init__()

        # Model parameters.
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_length = seq_length

        # LSTM Autoencoder.
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)

        # Linear layer to map hidden_dim to input_dim.
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        
        # Loss and optimizer.
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
    
    def class_name(self):

        # Return class name.
        return self.__class__.__name__
    
    def method_name(self):

        # Return caller's method name.
        return inspect.currentframe().f_back.f_code.co_name

    def error_line_number(self):

        tb = sys.exc_info()[2]
        if tb:
            return tb.tb_lineno

    def normalize(self, data, mean=None, std=None):
        if mean is None or std is None:
            mean = data.mean(axis=0)
            std = data.std(axis=0)
        return (data - mean) / std, mean, std

    def read_data(self):
        
        try:

            # Read dataset.
            dataset = pd.read_excel("apple_stock_data.xlsx")

            # Get normal data.
            normal_data = dataset[dataset['extreme_event'] == 0]
            X_normal = normal_data[['Open', 'High', 'Low', 'Close', 'Volume', 'daily_return']].to_numpy()
            X_normal, mean, std = self.normalize(X_normal)
            y_normal = normal_data[['extreme_event']].to_numpy()

            # Split normal data into training (70%) and validation (30%) sets.
            split_index = int(0.7 * len(X_normal))
            X_normal_train = X_normal[:split_index]
            y_normal_train = y_normal[:split_index]

            X_normal_val = X_normal[split_index:]
            y_normal_val = y_normal[split_index:]

            # Get anomalies.
            anomalies = dataset[dataset['extreme_event'] == 1]
            X_anomalies = anomalies[['Open', 'High', 'Low', 'Close', 'Volume', 'daily_return']].to_numpy()
            X_anomalies = (X_anomalies - mean) / std  # Normalize anomalies using training mean and std
            Y_anomalies = anomalies[['extreme_event']].to_numpy()

            # Print message.
            print(f"[green]Success[/green] -> {self.class_name()} > {self.method_name()}: Reading training, validation, test sets.")
        
        except Exception as e:

            # Print message.
            print(f"[red]Error[/red] -> {self.class_name()} > {self.method_name()}: {e}. Line number: {self.error_line_number()}")

            # Exit.
            raise SystemExit
        
        # Return data.
        return {
            "normal": {
                "train_features": X_normal_train,
                "train_target": y_normal_train,
                "val_features": X_normal_val,
                "val_target": y_normal_val
            },
            "anomalies": {
                "features": X_anomalies,
                "target": Y_anomalies
            }
        }

    def forward(self, x):

        # Encoder: Get hidden state and cell state
        _, (h, c) = self.encoder(x)

        # Prepare an initial input for decoder: Zeros of shape (batch_size, seq_length, hidden_dim)
        decoder_input = torch.zeros(x.size(0), x.size(1), self.hidden_dim).to(x.device)

        # Decoder: Use hidden state and cell state from encoder
        x_recon, _ = self.decoder(decoder_input, (h, c))

        # Map hidden_dim back to input_dim using the linear layer
        x_recon = self.output_layer(x_recon)

        return x_recon

    def create_sequences(self, features, targets):

        try:

            xs, ys = [], []
            for i in range(len(features)):
                # Extract a full sequence if possible.
                x = features[i:i + self.seq_length]
                y = features[i:i + self.seq_length]
                
                # If the sequence is less than seq_length, pad with zeros.
                if len(x) < self.seq_length:

                    # Pad the remaining length with zeros.
                    padding = np.zeros((self.seq_length - len(x), self.input_dim))
                    x = np.vstack((x, padding))
                    y = np.vstack((y, padding))
                
                # Only append if the sequence has the correct length.
                if len(x) == self.seq_length:
                    xs.append(x)
                    ys.append(y)

            print(f"[green]Success[/green] -> {self.class_name()} > {self.method_name()}. Total sequences created: {len(xs)}")

        except Exception as e:

            print(f"[red]Error[/red] -> {self.class_name()} > {self.method_name()}: {e}. Line number: {self.error_line_number()}")

        return np.array(xs), np.array(ys)

    def train_model(self, training_set = None, num_epochs = 50, bs = 64):
        try:
            # Prepare sequences.
            X_train_seq, y_train_seq = self.create_sequences(features = training_set['train_features'], targets = training_set['train_target'])

            # Convert to tensors.
            X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32)

            # Create DataLoader.
            train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=bs, shuffle=False)

            self.train()
            for epoch in range(num_epochs):
                total_loss = 0
                for X_batch, y_batch in train_loader:
                    
                    X_batch = X_batch.view(X_batch.size(0), X_batch.size(1), self.input_dim)

                    # Forward propagation.
                    self.optimizer.zero_grad()
                    X_pred = self(X_batch)
                    loss = self.criterion(X_pred, X_batch)
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()

                # Print progress.
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}")

            # Print message.
            print(f"[green]Success[/green] -> {self.class_name()} > {self.method_name()}: Model training.")
        
        except Exception as e:

            # Print message.
            print(f"[red]Error[/red] -> {self.class_name()} > {self.method_name()}: {e}. Line number: {self.error_line_number()}")

            # Exit.
            raise SystemExit
        
    def detect_anomalies(self, normal_set = None, anomaly_set = None, threshold_percentile = 95):
        try:
            # Step 1: Calculate reconstruction error for normal data to set threshold based on 20% of normal data out of 30%.
            X_normal_seq, _ = self.create_sequences(features=normal_set['val_features'], targets=normal_set['val_target'])
            X_normal_tensor = torch.tensor(X_normal_seq, dtype=torch.float32) # torch.Size([848, 10, 6])

            normal_errors = []
            self.eval()
            with torch.no_grad():
                for i in range(X_normal_tensor.size(0)):
                    X = X_normal_tensor[i].unsqueeze(0) # Adds a batch dimension at the beginning -> (1, 10, 6)
                    X_pred = self(X)
                    loss = self.criterion(X_pred, X)
                    normal_errors.append(loss.item())

            normal_errors = np.array(normal_errors)
            threshold = np.percentile(normal_errors, threshold_percentile)

            print(f"[green]Threshold set at {threshold:.4f} based on normal data.[/green]")

            # Step 2: Detect anomalies in anomalous data using the threshold.
            print("[green]Detecting anomalies in anomalous data...[/green]")
            X_anomaly_seq, y_anomaly_seq = self.create_sequences(features=anomaly_set['features'], targets=anomaly_set['target'])
            X_anomaly_tensor = torch.tensor(X_anomaly_seq, dtype=torch.float32)
           #y_anomaly_tensor = torch.tensor(y_anomaly_seq, dtype=torch.int32)

            anomaly_errors = []
            with torch.no_grad():
                for i in range(X_anomaly_tensor.size(0)):
                    X = X_anomaly_tensor[i].unsqueeze(0) #  Adds a batch dimension at the beginning. -> (1, 10, 6)
                    X_pred = self(X)
                    loss = self.criterion(X_pred, X)
                    anomaly_errors.append(loss.item())

            anomaly_errors = np.array(anomaly_errors)
            anomalies = np.where(anomaly_errors > threshold)[0]

            # Compare with true labels.
            #true_labels = X_anomaly_tensor[:, -1,0].numpy() # shape -> (457,)
            true_labels = torch.ones_like(X_anomaly_tensor[:, -1, 0]).cpu().numpy()  # Shape (457,)
            predicted_labels = np.zeros_like(true_labels)
            predicted_labels[anomalies] = 1  # Mark detected anomalies

            # Calculate performance metrics
            tp = np.sum((predicted_labels == 1) & (true_labels == 1))
            fn = np.sum((predicted_labels == 0) & (true_labels == 1))

            # Precision, Recall, and F1 Score
            precision = tp / (tp + 1e-6)  # Avoid division by zero
            recall = tp / (tp + fn + 1e-6)
            f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)

            # Print metrics
            print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")
            print(f"True Positives: {tp}, False Negatives: {fn}")
            print(f"Detected {len(anomalies)} anomalies out of {len(y_anomaly_seq)} samples.")

            # Plot errors and anomalies
            self.plot_errors(anomaly_errors, threshold, anomalies)

            # Print message.
            print(f"[green]Success[/green] -> {self.class_name()} > {self.method_name()}: Anomaly detection.")

        except Exception as e:

            # Print message.
            print(f"[red]Error[/red] -> {self.class_name()} > {self.method_name()}: {e}. Line number: {self.error_line_number()}")

            # Exit.
            raise SystemExit
        
        # Return data.
        return anomaly_errors, threshold, anomalies, precision, recall, f1_score

    def plot_errors(self, errors, threshold, anomalies):
        plt.figure(figsize=(10, 6))
        plt.plot(errors, label='Reconstruction Error')
        plt.axhline(y=threshold, color='r', linestyle='--', label='Anomaly Threshold')
        plt.scatter(anomalies, errors[anomalies], color='red', label='Anomalies')
        plt.xlabel('Batch Index')
        plt.ylabel('Error')
        plt.legend()
        plt.show()

    def workflow(self):

        # Read training, validation, testing sets.
        datasets_dict= self.read_data()

        # Train model.
        self.train_model(training_set = datasets_dict['normal'], num_epochs = 10, bs = 64)

        # Anomaly detection.
        # self.detect_anomalies(test_set=datasets_dict['anomalies'], threshold_percentile=95)
        self.detect_anomalies(normal_set=datasets_dict['normal'], anomaly_set=datasets_dict['anomalies'], threshold_percentile=95)

    @classmethod
    def run(cls):

        # Instantiate class object.
        instance = cls()

        # Run workflow.
        instance.workflow()
    
if __name__ == "__main__":

    LSTMAutoencoder.run()