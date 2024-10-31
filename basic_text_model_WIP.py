import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from preprocess_data import split_train_val_test
import matplotlib.pyplot as plt
import numpy as np
from settings import *


def remove_stop_words(input_string: str) -> str:
    """
    Returns given text but without stop words found in the czech_stopwords.txt file.
    Newlines are converted to spaces.
    """
    with open(CZECH_STOPWORDS, "r", encoding="utf-8") as f:
        stop_list = f.readlines()
    words = input_string.split()
    filtered_words = [word for word in words if word not in stop_list]
    output_string = ' '.join(filtered_words)
    return output_string


def balance_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Balance the dataset by randomly under-sampling class 1 to ensure each class has an equal amount of data.
    """
    only_1 = dataset[dataset["status"] == "1"]
    sample_size = len(dataset[dataset["status"] == "0"])
    random_same_length_only_1 = only_1.sample(n=sample_size)
    return pd.concat([random_same_length_only_1, dataset[dataset["status"] == "0"]])


def join_text_columns(dataset: pd.DataFrame, cols: list[str], separator="_") -> pd.Series:
    """
    Join text columns into one column. Texts are separated by separator.
    Useful for CountVectorizer as it accepts only one text column.
    """
    return dataset[cols].apply(lambda row: separator.join(row.values.astype(str)), axis=1)


def load_and_prepare_data() -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    lemmatized_df = pd.read_csv(DATA_DIR / "lemmatized_dataset.csv")
    preprocess_df = pd.read_csv(DATA_DIR / "paro_preprocessed.csv", usecols=["year"])

    lemmatized_df["project_name"] = lemmatized_df["project_name"].astype("string")
    lemmatized_df["project_description"] = lemmatized_df["project_description"].astype("string")
    lemmatized_df["public_interest"] = lemmatized_df["public_interest"].astype("string")
    lemmatized_df["year"] = preprocess_df["year"]

    df_train, df_val, _ = split_train_val_test(lemmatized_df)
    df_train = balance_dataset(df_train)

    df_train["status"] = df_train["status"].astype("int")
    df_val["status"] = df_val["status"].astype("int")

    cols = ["project_name", "project_description", "public_interest"]
    X_train = join_text_columns(df_train, cols=cols).apply(remove_stop_words)
    y_train = df_train["status"]

    X_val = join_text_columns(df_val, cols=cols).apply(remove_stop_words)
    y_val = df_val["status"]

    return X_train, y_train, X_val, y_val


if __name__ == "__main__":
    np.random.seed(42)
    X_train, y_train, X_val, y_val = load_and_prepare_data()

    pipeline = Pipeline([
            ('vectorizer', CountVectorizer()),
            ('classifier', LogisticRegression(max_iter=100, random_state=42)),
            # ('classifier', RandomForestClassifier(max_depth=4, n_estimators=30, random_state=42)),
            # ('classifier', RandomForestClassifier()),
        ])
    pipeline.fit(X_train, y_train)

    threshold = 0.5
    # you can see directly the predicted probabilities for each class
    predicted_proba = pipeline.predict_proba(X_val)
    # print("proba", predicted_proba[:50])

    y_pred = [0 if proba[0] >= threshold else 1 for proba in predicted_proba]

    print(classification_report(y_val, y_pred))
    print(ConfusionMatrixDisplay.from_predictions(y_val, y_pred))
    plt.show()
