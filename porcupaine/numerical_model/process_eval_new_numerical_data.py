import pandas as pd
import joblib
from porcupaine.settings import *
from porcupaine.preprocessing.preprocess_numerical_data import preprocess_data, transform_and_scale_budget


# Helper function: One-Hot Encoding with Alignment
def one_hot_encode_with_alignment(data, columns, training_columns):
    """
    Performs one-hot encoding on specified categorical columns and aligns the resulting columns
    with the columns used during model training.

    Args:
    - data (pd.DataFrame): The input DataFrame to encode.
    - columns (list): A list of column names to perform one-hot encoding on.
    - training_columns (list): The columns that were used during model training.

    Returns:
    - pd.DataFrame: The DataFrame with one-hot encoded columns, aligned with training columns.
    """
    # Apply one-hot encoding
    data = pd.get_dummies(data, columns=columns, dtype=int)

    # Reindex the columns to match the training columns (add missing columns as 0)
    data = data.reindex(columns=training_columns, fill_value=0)
    return data


# Main function: Predict Project Success
def predict_project_success(dataframe: pd.DataFrame, model_path):
    """
    Preprocesses the training data to get the columns, performs one-hot encoding for the new categorical columns and
    aligns it with the training data,transforms the 'budget' column, loads a pre-trained model, and predicts the
    probability of success for the new project.

    Args:
    - dataframe (pd.DataFrame): The new data for prediction.
    - model_path (str): Path to the saved logistic regression model.

    Returns:
    - predicted_success (float): The predicted probability of success for the new project.
    """

    # 4. Load the pre-trained model
    model = joblib.load(model_path)

    # Get training data columns (assuming preprocess_data is already run)
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(DATA_DIR / "paro_preprocessed.csv")

    # Apply one-hot encoding to the new data (category and district) and align it with the training columns
    dataframe = one_hot_encode_with_alignment(dataframe, ["category", "district"], X_train.columns)

    # 5. Transform and scale the "budget" column as in training
    dataframe["budget"] = dataframe["budget"].apply(transform_and_scale_budget)

    # 6. Make a prediction using the model
    prediction = model.predict_proba(dataframe)[:, 1]  # Get the probability of success (class 1)

    # Return the predicted probability of success
    return prediction[0]  # Return the prediction for the first (and only) project in the data


# Example usage:
if __name__ == "__main__":
    # Prepare the new data for prediction
    num_inputs = pd.DataFrame({
        'category': ["Zeleň"],  # Replace with your input values
        'district': ["Brno"],  # Replace with your input values
        'budget': [300000]  # Replace with your input values
    }, index=[0])

    # Define the path to the trained model
    model_path = MODELS_DIR / 'numerical_logistic_regression_model.pkl'

    # Make the prediction
    success_chance = predict_project_success(num_inputs, model_path)

    # Print the predicted probability of success
    print(f"The predicted chance of success for the new project is: {success_chance:.2f}")
