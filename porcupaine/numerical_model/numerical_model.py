import joblib  # For saving the model

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from porcupaine.preprocessing.preprocess_numerical_data import preprocess_data  # Preprocessing function
from porcupaine.settings import *


def preprocessing(file_path):
    """
    Preprocesses the dataset without scaling the features.
    Args:
        file_path (str): Path to the dataset file.
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    # Load and preprocess data (without scaling)
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(file_path)

    # No scaling here
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_logistic_regression(X_train, y_train):
    """
    Trains a logistic regression model on the provided training data.
    Args:
        X_train: Training features.
        y_train: Training labels.
    Returns:
        Trained model.
    """
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    return model


def save_model(model, file_name):
    """
    Saves the trained model to a file.
    Args:
        model: Trained logistic regression model.
        file_name (str): Path to save the model.
    """
    joblib.dump(model, file_name)
    print(f"Model saved to {file_name}")


def evaluate_model(model, X_val, y_val):
    """
    Evaluates the model only on the validation set.
    Args:
        model: Trained logistic regression model.
        X_val: Validation features.
        y_val: Validation labels.
    Returns:
        A dictionary containing evaluation metrics for the validation set.
    """
    # Predictions for validation set
    y_val_pred = model.predict(X_val)

    # Evaluation metrics for validation set
    evaluation_metrics = {
        "validation": {
            "accuracy": accuracy_score(y_val, y_val_pred),
            "precision": precision_score(y_val, y_val_pred),
            "recall": recall_score(y_val, y_val_pred),
            "f1": f1_score(y_val, y_val_pred),
            "roc_auc": roc_auc_score(y_val, y_val_pred),
        }
    }

    return evaluation_metrics, y_val_pred


def print_evaluation_results(evaluation_metrics):
    """
    Prints evaluation results for the validation set.
    Args:
        evaluation_metrics: Dictionary containing evaluation results for validation.
    """
    # Print evaluation results for validation set
    print("Validation set:")
    for metric, value in evaluation_metrics["validation"].items():
        print(f"{metric.capitalize()}: {value:.2f}")


def print_confusion_matrix(y_val, y_val_pred):
    """
    Prints the confusion matrix for the validation set.
    Args:
        y_val: True labels for validation set.
        y_val_pred: Predicted labels for validation set.
    """
    print("\nConfusion Matrix (Validation Set):")
    print(confusion_matrix(y_val, y_val_pred))


if __name__ == "__main__":
    # 1. Preprocess data without scaling
    file_path = "data/paro_preprocessed.csv"
    X_train, y_train, X_val, y_val, X_test, y_test = preprocessing(file_path)

    # 2. Train the Logistic Regression model
    model = train_logistic_regression(X_train, y_train)

    # 3. Evaluate the model (only on the validation set)
    evaluation_metrics, y_val_pred = evaluate_model(model, X_val, y_val)

    # 4. Print evaluation results
    print_evaluation_results(evaluation_metrics)

    # 5. Print confusion matrix for validation set
    print_confusion_matrix(y_val, y_val_pred)

    # 6. Save the trained model to a file
    save_model(model, MODELS_DIR / 'numerical_logistic_regression_model.pkl')
