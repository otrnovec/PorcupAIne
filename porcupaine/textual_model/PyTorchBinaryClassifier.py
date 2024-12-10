import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
import numpy as np
from abc import ABC, abstractmethod


def weighted_bce_loss(output, target, weight):
    bce_loss = nn.BCELoss(reduction="none")(output, target)
    return (bce_loss * weight).mean()


def compute_class_weights(labels):
    class_counts = torch.bincount(labels.long())
    class_weights = 1.0 / class_counts.float()
    return class_weights


class PyTorchBinaryClassifier(BaseEstimator, ClassifierMixin, ABC, nn.Module):
    """
    A wrapper of PyTorch binary classifier models for GridSearchCV from sklearn.
    Fit method has to be implemented in the child class according to the architecture used.
    """
    def __init__(self, lr=0.001, batch_size=32, epochs=5):
        """
        Initialize the PyTorchBinaryClassifier.
        Args:
            lr (float): Learning rate for the optimizer.
            batch_size (int): Batch size for training.
            epochs (int): Number of training epochs.
        """
        super().__init__()
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.classes_ = np.array([0, 1])      # attribute needed for GridSearchCV; TODO replace with not-hard-coded solution

    @abstractmethod         # TODO Is this declaration properly used? I hope so but I am not sure.
    def forward(self, x):
        """
        Forward pass for the model.
        Args:
            x (torch.Tensor): Input features.
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1) with probabilities.
        """
        pass

    def fit(self, X, y):
        """
        Train the PyTorch model.
        Args:
            X (pd.DataFrame): Training features.
            y (pd.DataFrame): Training labels.
        Returns:
            self: The fitted model.
        """
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        features = torch.tensor(X.values, dtype=torch.float32)
        labels = torch.tensor(y.values, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(features, labels)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        labels_tensor = labels.clone().detach().requires_grad_(True)
        class_weights = compute_class_weights(labels_tensor)

        for epoch in range(self.epochs):
            self.train()
            epoch_loss = 0
            for batch_features, batch_labels in dataloader:
                optimizer.zero_grad()

                outputs = self.forward(batch_features).squeeze()

                batch_class_weights = class_weights[batch_labels.long()]
                loss = weighted_bce_loss(outputs, batch_labels.float(), weight=batch_class_weights)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

    def predict(self, X):
        """
        Predict binary class labels.
        Args:
            X (pd.DataFrame): Input features.
        Returns:
            np.ndarray: Predicted labels.
        """
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X.values, dtype=torch.float32)
            outputs = self.forward(X_tensor)
        return (outputs.numpy() > 0.5).astype(int)

    def predict_proba(self, X):
        """
        Predict probabilities for binary classification.
        Args:
            X (pd.DataFrame): Input features.
        Returns:
            np.ndarray: Probabilities for the positive class.
        """
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X.values, dtype=torch.float32)
            outputs = self.forward(X_tensor)
        return outputs.numpy()

    def score(self, X, y, sample_weight=None):
        """
        Evaluate the accuracy of the model.

        Args:
            X (np.ndarray): Test features.
            y (np.ndarray): Test labels.
            sample_weight (np.ndarray, optional): Sample weights.

        Returns:
            float: Accuracy score.
        """
        y_pred = self.predict(X)
        if sample_weight is not None:
            return accuracy_score(y, y_pred, sample_weight=sample_weight)
        return accuracy_score(y, y_pred)
