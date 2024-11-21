import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_and_preprocess_data(file_name: str):

    # The function: loads data from an Excel file, removes unnecessary columns,
    # applies One-Hot Encoding, and scales the 'budget_millions' values.

    # 1. Loads the data from the .xlsx file
    data = pd.read_excel(file_name)

    # 2. Removes unwanted columns
    data = data.drop(columns=["row_number", "id"])

    # 3. One-Hot Encoding used on "project_category_codes" and "district_codes"
    data = pd.get_dummies(data, columns=["project_category_codes", "district_codes"])

    # 4. Converting the boolean values to integer (0 or 1)
    data = data.astype(int)

    # 5. Scaling the values in "budget_millions"
    scaler = MinMaxScaler()
    data[["budget_millions"]] = scaler.fit_transform(data[["budget_millions"]])

    return data


def save_data(data, output_file: str):

    # The function saves the processed data to an Excel file
    data.to_excel(output_file, index=False)
    print(f"Úpravy dat dokončeny a nový soubor byl uložen jako '{output_file}'.")


if __name__ == "__main__":
    # Define the input file and output file
    input_file = "numerical_model_dataset.xlsx"
    output_file = "upraveny_soubor_2.xlsx"

    # Load, preprocess, and save the data
    processed_data = load_and_preprocess_data(input_file)
    save_data(processed_data, output_file)
