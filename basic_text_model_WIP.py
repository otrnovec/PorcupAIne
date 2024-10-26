import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


lemmatized_df = pd.read_csv("data/lemmatized_df.csv", sep=";")

df_train, df_val, df_test = split_train_val_test(lemmatized_df)
X_train = df_train.drop(columns=["status"])
y_train = df_train["status"]

X_val = df_val.drop(columns=["status"])
y_val = df_val["status"]

pipeline = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', LogisticRegression(max_iter=1000)),
    ])
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_val)
print(classification_report(y_val, y_pred))

