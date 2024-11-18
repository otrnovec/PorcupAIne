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
from settings import *


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

            # Dynamically compute weights for the current batch
            batch_class_weights = class_weights[batch_labels.long()]
            loss = criterion(outputs, batch_labels, weight=batch_class_weights)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), "binary_classification_nn.pth")


def weighted_bce_loss(output, target, weight):
    bce_loss = nn.BCELoss(reduction="none")(output, target)
    return (bce_loss * weight).mean()


def evaluate_model(model: torch.nn.Module, X_val: pd.DataFrame, threshold:int=0.5) -> tuple[np.array, np.array]:
    """
    Evaluates the model on validation data and returns probabilities.

    Args:
        model (torch.nn.Module): Trained PyTorch model.
        X_val (pd.DataFrame): Validation feature set.
        threshold (int): threshold for probabilites to convert to 0 or 1


    Returns:
        tuple: Predicted probabilities and binary predictions
    """
    # Set the model to evaluation mode
    model.eval()

    features = torch.tensor(X_val.values, dtype=torch.float32)

    # Disable gradient computation for evaluation
    with torch.no_grad():
        # Get model predictions (sigmoid output for probabilities)
        probabilities = model(features).squeeze()

    binary_predictions = (probabilities >= threshold).float()

    return probabilities.numpy(), binary_predictions.numpy()


if __name__ == "__main__":
    path_to_embeddings = os.path.join(DATA_DIR, "contextual_embeddings.csv")
    X_train, y_train, X_val, y_val = get_train_val_data(path_to_embeddings, balanced=False)

    dataset = prepare_data(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    input_size = X_train.shape[1]       # number of columns
    model = BinaryClassificationNN(input_size)

    labels_tensor = torch.tensor(y_train.values, dtype=torch.long)
    class_weights = compute_class_weights(labels_tensor)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_model(model, dataloader, weighted_bce_loss, optimizer, class_weights, num_epochs=200)

    model.load_state_dict(torch.load("binary_classification_nn.pth", weights_only=True))
    probabilities, binary_predictions = evaluate_model(model, X_val)

    print(classification_report(y_val, binary_predictions))
    ConfusionMatrixDisplay.from_predictions(y_val, binary_predictions)
    plt.show()
