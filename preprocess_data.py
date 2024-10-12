import numpy as np
import pandas as pd
import matplotlib as plt

# where the dataset comes from
# https://data.brno.cz/search?collection=dataset&q=participativn%C3%AD%20rozpo%C4%8Det
df = pd.read_csv("data/PARO_original.csv", sep=";")

print(df.head())
print(df.info())

# drop unwanted columns
df = df.drop(columns=["properties.proposer", "properties.status.id", "properties.detail", "properties.image",
                      "ObjectId", "category.status", "category_category", "x", "y"])

df = df.rename(columns=lambda x: x.replace('properties.', ''))
df = df.rename(columns={"name": "project_name",
                        "category.name": "project_category",
                        "status.name": "status"
                        })

# drop unwanted rows
df = df[df["status"] != "jiný"]
df = df[df["status"] != "stažen"]
df = df[df["status"] != "nerealizovatelný"]
print(df["status"].value_counts())

print(df.info())

# split train-test-val datasets
df_train = df[(df["year"] != 2023) & (df["year"] != 2024)]
df_val = df[df["year"] == 2023]
df_test = df[df["year"] == 2024]


print(df_train["status"].value_counts())
print(df_val["status"].value_counts())
print(df_test["status"].value_counts())


# nevítězný           353   - jde provést, nevyhrál                         > feasible
# neproveditelný      281   - nejde provést, lidi o něm už nehlasují        > unfeasible
# nezískal podporu    231   - ???
# proveditelný         74   - 2024 proveditelný
# realizovaný          52   - proveditelný + lidi chtěli                    > wanted
# v realizaci          25   - proveditelný + lidi chtěli                    > wanted

# categorical to numeric
# df["status"].replace(inplace=True)
