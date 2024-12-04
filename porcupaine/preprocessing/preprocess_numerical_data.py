import pandas as pd

from porcupaine.settings import DATA_DIR


def preprocess_data(file_name):
    """
    Preprocesses the data from a CSV file by selecting relevant columns,
    transforming categorical columns, scaling numerical columns, and splitting
    the data into training, validation, and test sets based on the 'year' column.

    Args:
    - file_name (str): Path to the CSV file.

    Returns:
    - X_train, y_train, X_val, y_val, X_test, y_test: Processed datasets for training, validation, and testing.
    """
    # 1. Load data from the CSV file
    data = pd.read_csv(file_name)

    # 2. Select only relevant columns
    relevant_columns = ["project_category", "district", "budget", "status", "year"]
    data = data[relevant_columns]
    data.rename(columns={'project_category': 'category'}, inplace=True)
    data['district'] = data['district'].replace('Brno - Maloměřice A Obřany', 'Brno - Maloměřice a Obřany')

    # 3. Transform the "status" column into binary format
    def transform_status(value):
        positive_values = {"feasible", "unfeasible", "winning", "proveditelný"}
        return 1 if value in positive_values else 0

    data["status"] = data["status"].apply(transform_status)

    # 4. One-Hot Encoding for categorical columns with custom column naming
    def one_hot_encode_with_custom_names(data, columns):
        """
        Performs one-hot encoding on specified categorical columns and formats column names
        as `column_name_value`.

        Args:
        - data (pd.DataFrame): The input DataFrame to encode.
        - columns (list): A list of column names to perform one-hot encoding on.

        Returns:
        - pd.DataFrame: The DataFrame with one-hot encoded columns.
        """
        for column in columns:
            # Apply one-hot encoding for the column
            encoded = pd.get_dummies(data[column], prefix=column, prefix_sep="_", dtype=int)

            # Rename columns: drop the 'category_' part
            encoded.columns = [f"{column}_{col.split('_', 1)[-1]}" for col in encoded.columns]

            # Add the encoded columns to the DataFrame
            data = pd.concat([data, encoded], axis=1)

        # Drop the original categorical columns
        data = data.drop(columns=columns)

        return data

    data = one_hot_encode_with_custom_names(data, ["category", "district"])

    # 5. Transform and scale the "budget" column
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

    # 6. Split the data into training, validation, and test sets based on `year`
    # Training set: 2017–2022
    train_data = data[data["year"] <= 2022]
    # Validation set: 2023
    val_data = data[data["year"] == 2023]
    # Test set: 2024
    test_data = data[data["year"] == 2024]

    # 7. Remove the `year` column from each set
    train_data = train_data.drop(columns=["year"])
    val_data = val_data.drop(columns=["year"])
    test_data = test_data.drop(columns=["year"])

    # 8. Split into input features (X) and target variable (y)
    X_train = train_data.drop(columns=["status"])
    y_train = train_data["status"]

    X_val = val_data.drop(columns=["status"])
    y_val = val_data["status"]

    X_test = test_data.drop(columns=["status"])
    y_test = test_data["status"]

    # Return the processed datasets
    return X_train, y_train, X_val, y_val, X_test, y_test


# If this script is being run directly, you can test the function with a sample file
if __name__ == "__main__":
    # Example usage: Call the preprocess_data function and print the shapes of the datasets
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(DATA_DIR / "paro_preprocessed.csv")

    print(f"Training set: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"Validation set: X_val {X_val.shape}, y_val {y_val.shape}")
    print(f"Test set: X_test {X_test.shape}, y_test {y_test.shape}")

    # Save the processed training data to an Excel file
    #X_train.to_excel("processed_training_data.xlsx", index=False)
    #print("Processed training data saved to 'processed_training_data.xlsx'")
