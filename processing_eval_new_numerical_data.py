import pandas as pd
import joblib


def predict_project_success(file_name, model_path):
    """
    Preprocesses the input data, performs one-hot encoding for categorical columns,
    transforms the 'budget' column, loads a pre-trained model, and predicts the probability
    of success for the new project.

    Args:
    - file_name (str): Path to the new project data CSV file.
    - model_path (str): Path to the saved logistic regression model.

    Returns:
    - predicted_success (float): The predicted probability of success for the new project.
    """
    # 1. Load data from the CSV file
    data = pd.read_csv(file_name)

    # 2. Select only relevant columns
    relevant_columns = ["project_category", "district", "budget"]
    data = data[relevant_columns]

    # 3. One-Hot Encoding for categorical columns
    def one_hot_encode(data, columns):
        """
        Performs one-hot encoding on specified categorical columns in the DataFrame.

        Args:
        - data (pd.DataFrame): The input DataFrame to encode.
        - columns (list): A list of column names to perform one-hot encoding on.

        Returns:
        - pd.DataFrame: The DataFrame with one-hot encoded columns.
        """
        return pd.get_dummies(data, columns=columns, dtype=int)

    data = one_hot_encode(data, ["project_category", "district"])

    # 4. Transform and scale the "budget" column
    def transform_and_scale_budget(value):
        if value < 1000000:
            rounded_value = 1000000
        else:
            rounded_value = round(value, -6)  # Round to the nearest million

        scaling_map = {
            1000000: 0,
            2000000: 0.25,
            3000000: 0.50,
            4000000: 0.75,
            5000000: 1
        }
        return scaling_map.get(rounded_value, 1)

    data["budget"] = data["budget"].apply(transform_and_scale_budget)

    # 5. Load the pre-trained logistic regression model
    model = joblib.load(model_path)

    # 6. Make a prediction using the model
    prediction = model.predict_proba(data)[:, 1]  # Get the probability of success (class 1)

    # Return the predicted probability of success
    return prediction[0]  # Return the prediction for the first (and only) project in the data


# Example usage:
if __name__ == "__main__":
    # Example file path for the new project data
    file_path = "data/new_project.csv"

    # Path to the saved logistic regression model
    model_path = 'numerical_logistic_regression_model.pkl'

    # Call the function and get the predicted success chance
    success_chance = predict_project_success(file_path, model_path)

    print(f"The predicted chance of success for the new project is: {success_chance:.2f}")
