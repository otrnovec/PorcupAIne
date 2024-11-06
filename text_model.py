import pandas as pd
import os
import ast
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocess_data import split_train_val_test
from settings import *
from basic_text_model import balance_dataset


def process_embeddings(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Converts string representations of lists in specified columns of a DataFrame into actual lists,
    and then expands these lists into separate columns, with each element in the list becoming
    a new feature (column). The original columns are dropped after the expansion.
    :param df: The input DataFrame containing the columns with the embbedings in string representations of lists
    :param cols: A list of column names to be expanded into separate columns.
    :return: The DataFrame with the specified embedding columns expanded into multiple individual columns.
             Each new column represents one element from the original lists. The original embedding columns
             are removed from the resulting DataFrame.

    Example:
    If the input DataFrame `df` has a column 'project_name' with the following values:

    | project_name             | other_column |
    |--------------------------|--------------|
    | [0.1, 0.2, 0.3]          | "A"          |
    | [0.4, 0.5, 0.6]          | "B"          |

    After calling process_embeddings(df, ['project_name']), the result will be:

    | project_name_0 | project_name_1 | project_name_2 | other_column |
    |----------------|----------------|----------------|--------------|
    | 0.1            | 0.2            | 0.3            | "A"          |
    | 0.4            | 0.5            | 0.6            | "B"          |
    """
    for col in cols:
        df.loc[:, col] = df[col].apply(lambda x: ast.literal_eval(x))
        col_expanded = pd.DataFrame(df[col].tolist(), index=df.index)
        col_expanded.columns = [f"{col}_{i}" for i in range(col_expanded.shape[1])]
        df = pd.concat([df.drop(columns=[col]), col_expanded], axis=1)
    return df


if __name__ == "__main__":
    embedded_df = pd.read_csv(os.path.join(DATA_DIR, "embedded_dataset.csv"))
    preprocess_df = pd.read_csv(os.path.join(DATA_DIR, "paro_preprocessed.csv"), usecols=["year"])
    embedded_df["year"] = preprocess_df["year"]
    cols = ["project_name", "project_description", "public_interest"]

    df_train, df_val, _ = split_train_val_test(embedded_df)
    df_train = balance_dataset(df_train)

    X_train = process_embeddings(df_train[cols], cols)
    y_train = df_train["status"]
    X_val = process_embeddings(df_val[cols], cols)
    y_val = df_val["status"]

    # print(len(X_val.columns))     #
    pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('skb', SelectKBest(k=100)),
            ('classifier', RandomForestClassifier(max_depth=4, n_estimators=30, random_state=42))
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)

    print(classification_report(y_val, y_pred))
    ConfusionMatrixDisplay.from_predictions(y_val, y_pred)
    plt.show()
