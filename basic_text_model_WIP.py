import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer     # CountVectorizer accepts only one string column
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def prepare_data():
    lemmatized_df = pd.read_csv("data/lemmatized_df_small.csv", sep=",",
                                usecols=["project_name", "project_description", "public_interest", "status"])

    lemmatized_df["project_name"] = lemmatized_df["project_name"].astype("string")
    lemmatized_df["project_description"] = lemmatized_df["project_description"].astype("string")
    lemmatized_df["public_interest"] = lemmatized_df["public_interest"].astype("string")

    df_train, df_val = lemmatized_df[:65], lemmatized_df[65:]

    cols = ["project_name", "project_description", "public_interest"]
    X_train = df_train[cols].apply(
        lambda row: '_'.join(row.values.astype(str)), axis=1)
    y_train = df_train["status"]

    X_val = df_val[cols].apply(
        lambda row: '_'.join(row.values.astype(str)), axis=1)
    y_val = df_val["status"]

    return X_train, y_train, X_val, y_val


if __name__ == "__main__":
    X_train, y_train, X_val, y_val = prepare_data()

    pipeline = Pipeline([
            ('vectorizer', CountVectorizer()),
            # ('classifier', LogisticRegression(max_iter=100, random_state=42)),
            ('classifier', RandomForestClassifier(max_depth=4, n_estimators=30, random_state=42)),
        ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)
    print(classification_report(y_val, y_pred))

