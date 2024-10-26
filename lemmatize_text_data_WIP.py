from pathlib import Path
import pandas as pd
from preprocess_data import change_column_types, split_train_val_test
import requests
import json
from time import time


def get_lemmas(text: str) -> list:
    """
    for a given Czech text returns a list of lemmas
    calls NLP FI MUNI tagger api
    """
    data = {"call": "tagger",
            "lang": "cs",
            "output": "json",
            "text": text.replace(';', ',')     # very important!!! if semicolon not replaced by comma, json can't read it
            }
    uri = "https://nlp.fi.muni.cz/languageservices/service.py"
    response = requests.post(uri, data=data, timeout=10000)
    response.raise_for_status()  # raises exception when not a 2xx response
    byte_data = response.content
    data = json.loads(str(byte_data, 'utf-8'))
    tokens = [token for token in data["vertical"] if len(token) == 3]
    lemmas = []

    for token in tokens:
        lemmas.append(token[1])
    return lemmas


def get_lemmatized_column(column: pd.Series) -> pd.Series:
    """
    Returns the same pd.Series but with text as lemmas inside.
    Somewhat wrapper for get_lemmas function because NLP FI MUNI API does not allow more than 500 calls/day
    so the column has to be joint in one text and then split again and not send a cell by cell
    """
    joint_rows = ""
    for row in column:
        joint_rows += row + " XXX "

    lemmas = get_lemmas(joint_rows)

    lemmatized_row = ""
    lemmatized_rows = []
    for lemma in lemmas:
        if lemma != "XXX":
            lemmatized_row += lemma + " "
        else:
            lemmatized_rows.append(lemmatized_row)
            lemmatized_row = ""

    return pd.Series(lemmatized_rows)


def get_lemmatized_df(dataframe: pd.DataFrame) -> None:
    """
    creates a dataframe with lemmatized texts
    takes long time to execute!!! (around 20 minutes) because of the calls to external API with all text data
    it is split into so many parts because we want to avoid the HTTP run out of the time exception
    """
    lemmatized_df = pd.read_csv(r'C:\Users\haemk\myProject\data\lemmatized_df.csv')
    # lemmatized_df = pd.DataFrame()
    slice_value = len(dataframe["project_description"]) // 4

    # lemmatized_df["project_name"] = get_lemmatized_column(dataframe["project_name"])
    # print("1 done")
    # lemmatized_df.to_csv(Path(__file__).parent / "data" / "lemmatized_df.csv")

    # lemmatized_df["project_description"] =\
    #     get_lemmatized_column(dataframe["project_description"][:slice_value//2])
    # print("2 done")
    # lemmatized_df.to_csv(Path(__file__).parent / "data" / "lemmatized_df.csv")
    #
    # lemmatized_df["project_description"] = \
    #     get_lemmatized_column(dataframe["project_description"][slice_value // 2:slice_value])
    # print("2a done")
    # lemmatized_df.to_csv(Path(__file__).parent / "data" / "lemmatized_df.csv")
    # TODO correct the code so that it really concatenates together
    lemmatized_df["project_description"] = pd.concat([
        lemmatized_df["project_description"],
        get_lemmatized_column(dataframe["project_description"][slice_value:350])
    ], ignore_index=True)
    print("3 done")
    lemmatized_df.to_csv(Path(__file__).parent / "data" / "lemmatized_df.csv")
    lemmatized_df["project_description"] = pd.concat([
        lemmatized_df["project_description"],
        get_lemmatized_column(dataframe["project_description"][350:500])
    ], ignore_index=True)
    print("4 done")
    lemmatized_df.to_csv(Path(__file__).parent / "data" / "lemmatized_df.csv")

    lemmatized_df["project_description"] = pd.concat([
        lemmatized_df["project_description"],
        get_lemmatized_column(dataframe["project_description"][500:650])
    ], ignore_index=True)
    print("5 done")
    lemmatized_df.to_csv(Path(__file__).parent / "data" / "lemmatized_df.csv")

    lemmatized_df["project_description"] = pd.concat([
        lemmatized_df["project_description"],
        get_lemmatized_column(dataframe["project_description"][650:850])
    ], ignore_index=True)
    print("5a done")
    lemmatized_df.to_csv(Path(__file__).parent / "data" / "lemmatized_df.csv")

    lemmatized_df["project_description"] = pd.concat([
        lemmatized_df["project_description"],
        get_lemmatized_column(dataframe["project_description"][850:])
    ], ignore_index=True)
    print("5b done")
    lemmatized_df.to_csv(Path(__file__).parent / "data" / "lemmatized_df.csv")
    # -----------------------

    # lemmatized_df["public_interest"] = \
    #     get_lemmatized_column(dataframe["public_interest"][:slice_value])
    # print("6 done")
    # lemmatized_df.to_csv(Path(__file__).parent / "data" / "lemmatized_df.csv")

    lemmatized_df["public_interest"] = pd.concat([
        lemmatized_df["public_interest"],
        get_lemmatized_column(dataframe["public_interest"][slice_value:slice_value * 2])
    ], ignore_index=True)
    print("7 done")
    lemmatized_df.to_csv(Path(__file__).parent / "data" / "lemmatized_df.csv")

    lemmatized_df["public_interest"] = pd.concat([
        lemmatized_df["public_interest"],
        get_lemmatized_column(dataframe["public_interest"][slice_value * 2:slice_value * 3])
    ], ignore_index=True)
    print("8 done")
    lemmatized_df.to_csv(Path(__file__).parent / "data" / "lemmatized_df.csv")

    lemmatized_df["public_interest"] = pd.concat([
        lemmatized_df["public_interest"],
        get_lemmatized_column(dataframe["public_interest"][slice_value * 3:])
    ], ignore_index=True)
    print("9 done")
    lemmatized_df.to_csv(Path(__file__).parent / "data" / "lemmatized_df.csv")

    lemmatized_df["status"] = dataframe["status"]

    lemmatized_df.to_csv(Path(__file__).parent / "data" / "lemmatized_df.csv")


if __name__ == "__main__":
    df = pd.read_csv(Path(__file__).parent / "data" / "paro_preprocessed.csv")
    df = change_column_types(df)
    text_df = df[["project_name", "project_description", "public_interest", "status"]]
    text_df["status"].replace({
            "feasible": 1,
            "winning": 1,
            "unfeasible": 1,
            "without support": 0,
    }, inplace=True
    )
    start = time()
    get_lemmatized_df(text_df)
    end = time()
    print("celkový čas:", round(end-start, 2))
