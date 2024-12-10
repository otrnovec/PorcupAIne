"""
This file contains various neural network architectures inheriting from PyTorchBinaryClassifier.
"""
import torch.nn as nn

from porcupaine.textual_model.PyTorchBinaryClassifier import PyTorchBinaryClassifier


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
                 lstm_hidden_size=32, lstm_layers=1, dropout_rate=0):
        """
        Args:
            lr (float): Learning rate for the optimizer.
            batch_size (int): Batch size for training.
            epochs (int): Number of training epochs.
            lstm_hidden_size (int): Number of hidden units in each LSTM layer.
            lstm_layers (int): Number of stacked LSTM layers.
            dropout_rate (float): Dropout probability between LSTM layers.
        """
        super().__init__(lr, batch_size, epochs)
        self.input_size = input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.dropout_rate = dropout_rate

        self.params = {"lstm_hidden_size": lstm_hidden_size,
                       "lstm_layers": lstm_layers,
                       "dropout_rate": dropout_rate}

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_layers,
                            dropout=dropout_rate,
                            batch_first=True)

        self.fc = nn.Linear(lstm_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Reshape input for LSTM => Add feature dimension: (batch_size, seq_length, 1)
        x = x.unsqueeze(1)

        _, (h_n, _) = self.lstm(x)

        # Use the last hidden state from the top-most LSTM layer
        x = h_n[-1]  # (batch_size, lstm_hidden_size)

        x = self.fc(x)
        x = self.sigmoid(x)
        return x


class HybridCNNLSTM(PyTorchBinaryClassifier):
    def __init__(self,
                 lr=0.001, batch_size=32, epochs=5,
                 cnn_out_channels=16, kernel_size=3, lstm_hidden_size=32, lstm_layers=1, dropout_rate=0
                 ):
        super().__init__(lr, batch_size, epochs)
        self.cnn_out_channels = cnn_out_channels
        self.kernel_size = kernel_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.dropout_rate = dropout_rate

        self.params = {"cnn_out_channels": cnn_out_channels,
                       "kernel_size": kernel_size,
                       "lstm_hidden_size": lstm_hidden_size,
                       "lstm_layers": lstm_layers,
                       "dropout_rate": dropout_rate
                       }

        self.cnn = nn.Conv1d(in_channels=1,
                             out_channels=cnn_out_channels,
                             kernel_size=kernel_size,
                             stride=1,
                             padding=kernel_size // 2)
        self.relu = nn.ReLU()

        self.rnn = nn.LSTM(input_size=cnn_out_channels,
                           hidden_size=lstm_hidden_size,
                           num_layers=lstm_layers,
                           batch_first=True)

        self.dropout = nn.Dropout(p=dropout_rate)

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
