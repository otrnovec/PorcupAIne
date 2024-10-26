import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from pathlib import Path

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
    lemmatized_df = pd.read_csv(Path(__file__).parent / "data" / "lemmatized_df_small.csv", sep=",",
                                usecols=["project_name", "project_description", "public_interest", "status"])

    lemmatized_df["project_name"] = lemmatized_df["project_name"].astype("string")
    lemmatized_df["project_description"] = lemmatized_df["project_description"].astype("string")
    lemmatized_df["public_interest"] = lemmatized_df["public_interest"].astype("string")

    df_train, df_val = lemmatized_df[:65], lemmatized_df[65:]

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
    print(X_train[:5])
    pipeline = Pipeline([
            ('vectorizer', CountVectorizer()),
            # ('classifier', LogisticRegression(max_iter=100, random_state=42)),
            ('classifier', RandomForestClassifier(max_depth=4, n_estimators=30, random_state=42)),
        ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)
    print(classification_report(y_val, y_pred))

