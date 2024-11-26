import pandas as pd
import joblib

from porcupaine.settings import *


def predict_project_success(dataframe: pd.DataFrame, model_path):
    """
    Preprocesses the input data, performs one-hot encoding for categorical columns,
    transforms the 'budget' column, loads a pre-trained model, and predicts the probability
    of success for the new project.

    Args:
    - datafrane (pd.DataFrame): object with new
    - model_path (str): Path to the saved logistic regression model.

    Returns:
    - predicted_success (float): The predicted probability of success for the new project.
    """

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

    dataframe = one_hot_encode(dataframe, ["category", "district"])

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

    dataframe["budget"] = dataframe["budget"].apply(transform_and_scale_budget)

    # 5. Load the pre-trained logistic regression model
    model = joblib.load(model_path)

    # 6. Make a prediction using the model
    prediction = model.predict_proba(dataframe)[:, 1]  # Get the probability of success (class 1)

    # Return the predicted probability of success
    return prediction[0]  # Return the prediction for the first (and only) project in the data


def demo_predict_project_success(project_category, district, budget):

    data = {
    'project_category': [project_category],
    'district': [district],
    'budget': [budget]
    }

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    df.to_csv(DATA_DIR / 'demo_project.csv', index=False)

    predict_project_success(DATA_DIR/ "demo_project.csv", MODELS_DIR / "numerical_logistic_regression_model.pkl")


# Example usage:
if __name__ == "__main__":
    num_inputs = pd.DataFrame({
        'category': "Senioři",
        'district': "Brno - Bohunice",
        'budget': 2500000
    }, index=[0])

    model_path = MODELS_DIR / 'numerical_logistic_regression_model.pkl'

    success_chance = predict_project_success(num_inputs, model_path)
    print(f"The predicted chance of success for the new project is: {success_chance:.2f}")

