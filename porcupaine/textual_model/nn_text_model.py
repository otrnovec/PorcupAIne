import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt

from text_model import get_train_val_data
from porcupaine.settings import *


class BinaryClassificationNN(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassificationNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)


class HybridCNNRNN(nn.Module):
    """
    A hybrid CNN-RNN model for binary classification.
    """

    def __init__(self, cnn_out_channels=16, kernel_size=3, rnn_hidden_size=32, rnn_layers=1):
        super(HybridCNNRNN, self).__init__()

        # Convolutional Layer
        self.cnn = nn.Conv1d(in_channels=1,
                             out_channels=cnn_out_channels,
                             kernel_size=kernel_size,
                             stride=1,
                             padding=kernel_size // 2)
        self.relu = nn.ReLU()

        # Recurrent Layer (LSTM)
        self.rnn = nn.LSTM(input_size=cnn_out_channels,
                           hidden_size=rnn_hidden_size,
                           num_layers=rnn_layers,
                           batch_first=True)

        self.dropout = nn.Dropout(p=0.3)

        # Fully connected layer for binary classification
        self.fc = nn.Linear(rnn_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass through the hybrid CNN-RNN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1) with probabilities.
        """
        # Reshape input to match Conv1D requirements: (batch_size, channels, input_size)
        x = x.unsqueeze(1)  # Add channel dimension

        # Pass through CNN
        x = self.cnn(x)
        x = self.relu(x)

        # Transpose for RNN: (batch_size, seq_length, cnn_out_channels)
        x = x.transpose(1, 2)

        # Pass through RNN
        _, (h_n, _) = self.rnn(x)

        # Take the last hidden state and pass through the fully connected layer
        x = h_n[-1]  # (batch_size, rnn_hidden_size)
        x = self.fc(x)

        # Apply sigmoid for binary classification
        x = self.sigmoid(x)

        return x


class EnhancedLSTM(nn.Module):
    """
    @todo The class does not work now. There is a problem with tensor dimensions.
    Enhanced LSTM model for binary classification with multiple LSTM layers and dropout between layers.

    The input is a vector, and the output is a tensor of shape (batch_size, output_size).
    """

    def __init__(self, input_size, hidden_size=32, num_layers=3, output_size=1, dropout_prob=0.3):
        """
        Initializes the LSTM model.

        Args:
            input_size (int): The number of features in the input vector.
            hidden_size (int, optional): The number of hidden units in each LSTM layer (default is 32).
            num_layers (int, optional): The number of LSTM layers (default is 3).
            output_size (int, optional): The number of output units (default is 1 for binary classification).
            dropout_prob (float, optional): The probability of dropout (default is 0.3).
        """
        super(EnhancedLSTM, self).__init__()

        # LSTM layer with dropout between layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout_prob, bidirectional=False)

        # Fully connected layer for binary classification
        self.fc = nn.Linear(hidden_size, output_size)

        # Sigmoid activation for binary output
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass for the LSTM model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            Tensor: Output tensor of shape (batch_size, output_size).
        """
        # Pass through LSTM layers
        _, (lstm_out, _) = self.lstm(x)  # lstm_out shape: (batch_size, seq_length, hidden_size)

        # Get the last output of the LSTM for each sequence in the batch
        last_hidden_state = lstm_out[-1, :]  # shape: (batch_size, hidden_size)

        # Pass through the fully connected layer
        output = self.fc(last_hidden_state)  # shape: (batch_size, output_size)

        # Apply sigmoid activation
        output = self.sigmoid(output)  # shape: (batch_size, output_size)

        return output


def prepare_data(X_dataframe, y_dataframe):
    features = torch.tensor(X_dataframe.values, dtype=torch.float32)
    labels = torch.tensor(y_dataframe.values, dtype=torch.float32)  # Binary labels
    return TensorDataset(features, labels)


def compute_class_weights(labels):
    class_counts = torch.bincount(labels.long())
    class_weights = 1.0 / class_counts.float()
    return class_weights


def train_model(model, dataloader, criterion, optimizer, class_weights, num_epochs=10):
    """
    Returns: None, but saves the model
    """
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch_features, batch_labels in dataloader:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_features).squeeze()
            # outputs = model(batch_features)
            # print(batch_labels)
            # print(outputs)
            # Dynamically compute weights for the current batch
            batch_class_weights = class_weights[batch_labels.long()]
            loss = criterion(outputs, batch_labels.float(), weight=batch_class_weights)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), "binary_classification_nn.pth")


def weighted_bce_loss(output, target, weight):
    bce_loss = nn.BCELoss(reduction="none")(output, target)
    return (bce_loss * weight).mean()


def get_predictions(model: torch.nn.Module, features: pd.DataFrame, threshold:int=0.5) -> tuple[np.array, np.array]:
    """
    Returns predictions for given dataset of features. Suitable for Torch NN models.

    Args:
        model (torch.nn.Module): Trained PyTorch model.
        features (pd.DataFrame): Feature set.
        threshold (int): threshold for probabilites to convert to 0 or 1

    Returns:
        tuple [np.array, np.array]: Predicted probabilities and binary predictions
    """
    # Set the model to evaluation mode
    model.eval()

    features = torch.tensor(features.values, dtype=torch.float32)

    # Disable gradient computation for evaluation
    with torch.no_grad():
        # Get model predictions (sigmoid output for probabilities)
        probabilities = model(features).squeeze()

    binary_predictions = (probabilities >= threshold).float()

    return probabilities.numpy(), binary_predictions.numpy()


if __name__ == "__main__":
    path_to_embeddings = DATA_DIR / "contextual_embeddings.csv"
    X_train, y_train, X_val, y_val = get_train_val_data(path_to_embeddings, balanced=False)

    dataset = prepare_data(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    input_size = X_train.shape[1]       # number of columns
    model = BinaryClassificationNN(input_size)
    # model = HybridCNNRNN(rnn_layers=3)
    # model = EnhancedLSTM(input_size)

    labels_tensor = torch.tensor(y_train.values, dtype=torch.long)
    class_weights = compute_class_weights(labels_tensor)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_model(model, dataloader, weighted_bce_loss, optimizer, class_weights, num_epochs=20)

    model.load_state_dict(torch.load("binary_classification_nn.pth", weights_only=True))
    probabilities, binary_predictions = get_predictions(model, X_val)

    print(classification_report(y_val, binary_predictions))
    ConfusionMatrixDisplay.from_predictions(y_val, binary_predictions)
    plt.show()
