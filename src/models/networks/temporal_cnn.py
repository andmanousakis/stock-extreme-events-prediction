import torch
import torch.nn as nn

class TemporalCNNNetwork(nn.Module):
    """
    Temporal CNN: two 1D convolutions over the days of a window, then a dense layer scoring no event vs extreme event.
    """

    def __init__(self, num_features: int = 6, window: int = 10, channels: list = None, kernel_size: int = 3, dropout: float = 0.3):
        """
        Build the layers.

        Args:
            num_features (int): Number of input features per day.
            window (int): Number of days per window.
            channels (list): Output channels of the two convolution layers, e.g. [32, 64].
            kernel_size (int): Number of days each convolution filter looks at.
            dropout (float): Share of activations dropped during training after each convolution.
        """

        # Set up the module.
        super().__init__()

        # Default to 32 then 64 channels.
        channels = channels or [32, 64]

        # First convolution over the days, keeping the window length.
        self.conv1 = nn.Conv1d(in_channels = num_features, out_channels = channels[0], kernel_size = kernel_size, padding = kernel_size // 2)

        # Second convolution over the days, keeping the window length.
        self.conv2 = nn.Conv1d(in_channels = channels[0], out_channels = channels[1], kernel_size = kernel_size, padding = kernel_size // 2)

        # ReLU activation after each convolution.
        self.relu = nn.ReLU()

        # Dropout after each convolution to reduce overfitting.
        self.dropout = nn.Dropout(dropout)

        # Flatten channels and days into one vector.
        self.flatten = nn.Flatten()

        # Dense layer scoring the two classes.
        self.fc = nn.Linear(channels[1] * window, 2)

    def forward(self, x: torch.Tensor = None) -> torch.Tensor:
        """
        Score a batch of windows. Returns raw scores (logits), since cross-entropy loss applies softmax itself.

        Args:
            x (torch.Tensor): Windows shaped [batch, features, days].

        Returns:
            torch.Tensor: Class scores shaped [batch, 2].
        """

        # First convolution, ReLU and dropout.
        x = self.dropout(self.relu(self.conv1(x)))

        # Second convolution, ReLU and dropout.
        x = self.dropout(self.relu(self.conv2(x)))

        # Flatten and score the classes.
        return self.fc(self.flatten(x))

    def predict_proba(self, x: torch.Tensor = None) -> torch.Tensor:
        """
        Predict class probabilities with softmax, without dropout or gradients.

        Args:
            x (torch.Tensor): Windows shaped [batch, features, days].

        Returns:
            torch.Tensor: Probabilities of no event and extreme event, shaped [batch, 2].
        """

        # Switch off dropout.
        self.eval()

        # Predict without tracking gradients.
        with torch.no_grad():

            # Apply softmax to the class scores.
            return torch.softmax(self.forward(x), dim = 1)
