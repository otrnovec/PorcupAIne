import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from pathlib import Path
from preprocess_data import split_train_val_test
import matplotlib.pyplot as plt
import numpy as np

czech_stop_list = [
    "být", "v", "a", "sebe", "na", "ten", "s", "z", "že", "který", "o", "mít", "i", "do", "on", "k",
    "pro", "tento", "za", "by", "moci", "svůj", "ale", "po", "rok", "jako", "však", "od", "všechen",
    "dva", "nebo", "tak", "u", "při", "jeden", "podle", "Praha", "jen", "další", "jeho", "aby", "co",
    "český", "jak", "veliký", "nový", "až", "už", "muset", "než", "nebýt", "člověk", "jenž", "léto",
    "firma", "první", "náš", "také", "my", "jejich", "když", "před", "doba", "chtít", "jiný", "mezi",
    "ještě", "já", "ani", "cena", "již", "jít", "strana", "či", "druhý", "pouze"
]


def remove_stop_words(input_string: str, stop_list: list = czech_stop_list) -> str:
    words = input_string.split()
    filtered_words = [word for word in words if word not in stop_list]
    output_string = ' '.join(filtered_words)
    return output_string


def load_and_split_data() -> tuple:
    lemmatized_df = pd.read_csv(Path(__file__).parent / "data" / "lemmatized_dataset.csv")
    preprocess_df = pd.read_csv(Path(__file__).parent / "data" / "paro_preprocessed.csv")

    lemmatized_df["project_name"] = lemmatized_df["project_name"].astype("string")
    lemmatized_df["project_description"] = lemmatized_df["project_description"].astype("string")
    lemmatized_df["public_interest"] = lemmatized_df["public_interest"].astype("string")
    lemmatized_df["year"] = preprocess_df["year"]

    df_train, df_val, _ = split_train_val_test(lemmatized_df)

    df_train["status"] = df_train["status"].astype("int")
    df_val["status"] = df_val["status"].astype("int")

    # text columns are joint together because CountVectorizer accepts only one string column
    cols = ["project_name", "project_description", "public_interest"]
    X_train = df_train[cols].apply(
        lambda row: '_'.join(row.values.astype(str)), axis=1)
    X_train = X_train.apply(remove_stop_words)
    y_train = df_train["status"]

    X_val = df_val[cols].apply(
        lambda row: '_'.join(row.values.astype(str)), axis=1)
    X_val = X_val.apply(remove_stop_words)
    y_val = df_val["status"]

    return X_train, y_train, X_val, y_val


if __name__ == "__main__":
    X_train, y_train, X_val, y_val = load_and_split_data()

    # under sampling class 1 to have equal size as class 0
    only_1 = y_train[y_train == 1]
    sample_size = len(y_train[y_train == 0])
    random_same_length_only_1 = np.random.choice(only_1, sample_size, replace=False)

    pipeline = Pipeline([
            ('vectorizer', CountVectorizer()),
            # ('classifier', LogisticRegression(max_iter=100, random_state=42)),
            # ('classifier', RandomForestClassifier(max_depth=4, n_estimators=30, random_state=42)),
            ('classifier', RandomForestClassifier()),
        ])
    pipeline.fit(X_train, y_train)

    threshold = 0.5
    # you can see directly the predicted probabilities for each class
    predicted_proba = pipeline.predict_proba(X_val)
    print("proba", predicted_proba[:50])

    # need to change the types due to the adjustments made to the classification threshold
    y_pred = []
    for proba in predicted_proba:
        if proba[0] >= threshold:
            y_pred.append(int(0))
        else:
            y_pred.append(int(1))

    print(classification_report(y_val, y_pred))
    print(ConfusionMatrixDisplay.from_predictions(y_val, y_pred))
    plt.show()
