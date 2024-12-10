"""
This file contains various neural network architectures inheriting from PyTorchBinaryClassifier.
"""
import torch.nn as nn
from sklearn.model_selection import GridSearchCV

from porcupaine.settings import *
from porcupaine.textual_model.PyTorchBinaryClassifier import PyTorchBinaryClassifier
from porcupaine.textual_model.text_model import get_train_val_data


class MultiLayerPerceptronNN(PyTorchBinaryClassifier):
    def __init__(self,
                 input_size,
                 lr=0.001, batch_size=32, epochs=5
                 ):
        super().__init__(lr, batch_size, epochs)
        self.input_size = input_size
        self.fc = nn.Sequential(                # fc means fully connected
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


class MultiLSTMBinaryClassifier(PyTorchBinaryClassifier):
    def __init__(self, input_size,
                 lr=0.001, batch_size=32, epochs=5,
                 lstm_hidden_size=32, lstm_layers=2, dropout=0.3):
        """
        Args:
            lr (float): Learning rate for the optimizer.
            batch_size (int): Batch size for training.
            epochs (int): Number of training epochs.
            lstm_hidden_size (int): Number of hidden units in each LSTM layer.
            lstm_layers (int): Number of stacked LSTM layers.
            dropout (float): Dropout probability between LSTM layers.
        """
        super().__init__(lr, batch_size, epochs)
        self.input_size = input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.dropout = dropout

        # LSTM Layers
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_layers,
                            dropout=dropout,
                            batch_first=True)

        self.fc = nn.Linear(lstm_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Reshape input for LSTM => Add feature dimension: (batch_size, seq_length, 1)
        x = x.unsqueeze(1)

        # Pass through LSTM layers
        _, (h_n, _) = self.lstm(x)

        # Use the last hidden state from the top-most LSTM layer
        x = h_n[-1]  # (batch_size, lstm_hidden_size)

        x = self.fc(x)
        x = self.sigmoid(x)
        return x


class HybridCNNLSTM(PyTorchBinaryClassifier):
    def __init__(self,
                 lr=0.001, batch_size=32, epochs=5,
                 cnn_out_channels=16, kernel_size=3, lstm_hidden_size=32, lstm_layers=1
                 ):
        super().__init__(lr, batch_size, epochs)
        self.cnn_out_channels = cnn_out_channels
        self.kernel_size = kernel_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers

        # Convolutional Layer
        self.cnn = nn.Conv1d(in_channels=1,
                             out_channels=cnn_out_channels,
                             kernel_size=kernel_size,
                             stride=1,
                             padding=kernel_size // 2)
        self.relu = nn.ReLU()

        # Recurrent Layer (LSTM)
        self.rnn = nn.LSTM(input_size=cnn_out_channels,
                           hidden_size=lstm_hidden_size,
                           num_layers=lstm_layers,
                           batch_first=True)

        self.dropout = nn.Dropout(p=0.3)

        # Fully connected layer for binary classification
        self.fc = nn.Linear(lstm_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Reshape input to match Conv1D requirements: (batch_size, channels, input_size) => Add channel dimension
        x = x.unsqueeze(1)

        x = self.cnn(x)
        x = self.relu(x)

        # Transpose for RNN: (batch_size, seq_length, cnn_out_channels)
        x = x.transpose(1, 2)

        _, (h_n, _) = self.rnn(x)

        # Take the last hidden state and pass through the fully connected layer
        x = h_n[-1]  # (batch_size, rnn_hidden_size)
        x = self.fc(x)
        x = self.sigmoid(x)

        return x


if __name__ == "__main__":
    path_to_embeddings = DATA_DIR / "contextual_embeddings.csv"
    X_train, y_train, X_val, y_val = get_train_val_data(path_to_embeddings, balanced=False)
    # Define parameter grid
    # param_grid = {
    #     'lr': [0.001, 0.01],
    #     'batch_size': [16, 32],
    #     'epochs': [5, 10, 20],
    # }
    param_grid = {
        'lr': [0.01],
        'batch_size': [32],
        'epochs': [5],
    }

    # Wrap PyTorch model
    # pytorch_clf = MultiLayerPerceptronNN(X_train.shape[1])
    # pytorch_clf = HybridCNNLSTM()
    pytorch_clf = MultiLSTMBinaryClassifier(X_train.shape[1])

    # Perform grid search
    grid_search = GridSearchCV(estimator=pytorch_clf, param_grid=param_grid, scoring='accuracy', cv=3)
    grid_search.fit(X_train, y_train)

    # Display the best parameters and accuracy
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validated accuracy:", grid_search.best_score_)

    # Test set evaluation
    best_model = grid_search.best_estimator_
    test_score = best_model.score(X_val, y_val)
    print("Test set accuracy:", test_score)


