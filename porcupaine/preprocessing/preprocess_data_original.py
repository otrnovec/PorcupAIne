"""
Preprocess the original PaRo dataset from here
        https://data.brno.cz/search?collection=dataset&q=participativn%C3%AD%20rozpo%C4%8Det
into a paro_preprocess.csv file
"""
import pandas as pd
from pathlib import Path


def split_train_val_test(dataframe):
    """
    Splits the PaRo dataset based on years.
    Train data are from 2017 to 2022, validation data are from 2023 and test data are from 2024
    @returns: train, val, test dataset in this order
    """
    return dataframe[(dataframe["year"] != 2023) & (dataframe["year"] != 2024)], dataframe[dataframe["year"] == 2023], dataframe[dataframe["year"] == 2024]


def change_column_types(dataframe):
    """
    Changes the column types in PaRo dataset in order to make them more computer-readable
    """
    dataframe["project_name"] = dataframe["project_name"].astype("string")

    dataframe["latitude"] = dataframe["latitude"].replace({",": "."}, regex=True)
    dataframe["longitude"] = dataframe["longitude"].replace({",": "."}, regex=True)
    dataframe["latitude"] = dataframe["latitude"].astype("float")
    dataframe["longitude"] = dataframe["longitude"].astype("float")

    dataframe["project_description"] = dataframe["project_description"].astype("string")
    dataframe["public_interest"] = dataframe["public_interest"].astype("string")

    dataframe["project_category"] = dataframe["project_category"].astype("category")
    dataframe["status"] = dataframe["status"].astype("category")
    dataframe["district"] = dataframe["district"].astype("category")

    dataframe["project_category_codes"] = dataframe["project_category"].cat.codes
    dataframe["status_codes"] = dataframe["status"].cat.codes
    dataframe["district_codes"] = dataframe["district"].cat.codes
    return dataframe


if __name__ == "__main__":
    # where the dataset comes from
    #
    df = pd.read_csv("../data/PARO_original.csv", sep=";")

    # print(df.head())
    # print(df.info())

    # drop unwanted columns
    df = df.drop(columns=["properties.proposer", "properties.status.id", "properties.detail", "properties.image",
                          "ObjectId", "category.status", "category_category", "x", "y"])

    # rename columns
    df = df.rename(columns=lambda x: x.replace('properties.', ''))
    df = df.rename(columns={"name": "project_name",
                            "category.name": "project_category",
                            "status.name": "status"
                            })

    # merge with descriptions nad interests
    descriptions_interests_df = pd.read_csv("../data/descriptions_and_interests.csv", sep=";")
    descriptions_interests_df = descriptions_interests_df.rename(columns={"project_id": "id"})
    df = pd.merge(df, descriptions_interests_df, on="id")
    df = df.drop(columns=["Unnamed: 0"])

    df = change_column_types(df)

    # drop unwanted rows
    df = df[df["status"] != "jiný"]
    df = df[df["status"] != "stažen"]
    df = df[df["status"] != "nerealizovatelný"]
    print(df["status"].value_counts())

    # rename outputs based on this schema:
    # nevítězný           353   - s podporou, proveditelný, lidi nechtěli                   > feasible
    # neproveditelný      281   - s podporou, neproveditelný, lidi o něm už nehlasují       > unfeasible
    # nezískal podporu    231   - ani se nezjišťovala proveditelnost                        > without support
    # proveditelný         74   - 2024 proveditelný
    # realizovaný          52   - proveditelný + lidi chtěli                                > winning
    # v realizaci          25   - proveditelný + lidi chtěli                                > winning
    df["status"].replace({
        "nevítězný": "feasible",
        "neproveditelný": "unfeasible",
        "nezískal podporu": "without support",
        "realizovaný": "winning",
        "v realizaci": "winning"
    }, inplace=True
    )

    df.to_csv(Path(__file__).parent / "data" / "paro_preprocessed.csv")
