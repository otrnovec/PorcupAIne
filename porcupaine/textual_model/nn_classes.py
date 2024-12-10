import numpy as np
import torch.nn as nn
from sklearn.model_selection import GridSearchCV, train_test_split

from porcupaine.textual_model.PyTorchBinaryClassifier import PyTorchBinaryClassifier


class MultiLayerPerceptronNN(PyTorchBinaryClassifier):
    def __init__(self, input_size, lr, batch_size, epochs):
        super().__init__(lr, batch_size, epochs)
        self.input_size = input_size
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

    # implementing abstract method
    def forward(self, x):
        return self.fc(x)


if __name__ == "__main__":

    # Generate sample data
    X, y = np.random.rand(1000, 10), np.random.randint(0, 2, 1000)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define parameter grid
    param_grid = {
        'lr': [0.001, 0.01],
        'batch_size': [16, 32],
        'epochs': [10, 20],
    }

    # Wrap PyTorch model
    pytorch_clf = MultiLayerPerceptronNN(X.shape[1], 0.001, 32, 5)

    # Perform grid search
    grid_search = GridSearchCV(estimator=pytorch_clf, param_grid=param_grid, scoring='accuracy', cv=3)
    grid_search.fit(X_train, y_train)

    # Display the best parameters and accuracy
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validated accuracy:", grid_search.best_score_)

    # Test set evaluation
    best_model = grid_search.best_estimator_
    test_score = best_model.score(X_test, y_test)
    print("Test set accuracy:", test_score)


