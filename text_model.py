import pandas as pd
import os
import ast
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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


def find_best_params(input_data, output_data) -> dict:
    """
        Performs a grid search to find the best hyperparameters for a pipeline consisting
        of MinMaxScaler, SelectKBest, and a RandomForestClassifier.
        :param input_data: Features for model training.
        :param output_data: Target variable for model training.
        :return Best hyperparameters found during the grid search.
    """
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('skb', SelectKBest()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    # Parameters of pipelines can be set using '__' separated parameter names:
    param_grid = {
        "skb__k": [50, 100, 500, 1000],
        "skb__score_func": [chi2, f_classif],
        "classifier__max_depth": [2, 4, 6, 20],
        "classifier__n_estimators": [20, 50, 100, 200, 500]
    }

    # n_jobs=-1 means the work is parallelized - all processors are used
    search = GridSearchCV(pipeline, param_grid, n_jobs=-1)

    search.fit(input_data, output_data)
    return search.best_params_


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

    # print(len(X_val.columns))     # there are 2304 vectors
    # print(find_best_params(X_train, y_train))

    # using best params find by GridSearchCV
    pipeline = Pipeline([
            # if we use MinMax instead of Standard scaler there is no problem with negative values while using chi2
            # however it performs differently (don't know exactly how...), so be careful:)
            ('scaler', MinMaxScaler()),
            # ('scaler', StandardScaler()),
            ('skb', SelectKBest(score_func=chi2, k=500)),
            ('classifier', RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)

    print(classification_report(y_val, y_pred))
    ConfusionMatrixDisplay.from_predictions(y_val, y_pred)
    plt.show()
