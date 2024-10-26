import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# where the dataset comes from
# https://data.brno.cz/search?collection=dataset&q=participativn%C3%AD%20rozpo%C4%8Det
df = pd.read_csv("data/PARO_original.csv", sep=";")

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
descriptions_interests_df = pd.read_csv("data/descriptions_and_interests.csv", sep=";")
descriptions_interests_df = descriptions_interests_df.rename(columns={"project_id": "id"})
df = pd.merge(df, descriptions_interests_df, on="id")
df = df.drop(columns=["Unnamed: 0"])

# change column types
df["project_name"] = df["project_name"].astype("string")

df["latitude"] = df["latitude"].replace({",": "."}, regex=True)
df["longitude"] = df["longitude"].replace({",": "."}, regex=True)
df["latitude"] = df["latitude"].astype("float")
df["longitude"] = df["longitude"].astype("float")

df["project_description"] = df["project_description"].astype("string")
df["public_interest"] = df["public_interest"].astype("string")

df["project_category"] = df["project_category"].astype("category")
df["status"] = df["status"].astype("category")
df["district"] = df["district"].astype("category")

df["project_category_codes"] = df["project_category"].cat.codes
df["status_codes"] = df["status"].cat.codes
df["district_codes"] = df["district"].cat.codes

# drop unwanted rows
df = df[df["status"] != "jiný"]
df = df[df["status"] != "stažen"]
df = df[df["status"] != "nerealizovatelný"]
print(df["status"].value_counts())

# rename outputs based on this:
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

# split train-test-val datasets
df_train = df[(df["year"] != 2023) & (df["year"] != 2024)]
df_val = df[df["year"] == 2023]
df_test = df[df["year"] == 2024]

# ---------------------------------------------------

print(df.info())

print(df_train["status"].value_counts())
print(df_val["status"].value_counts())
print(type(df_test["status"].value_counts()))

# ax = df["year"].value_counts().sort_values(ascending=False).plot.bar()
# plt.title("How many projects was submitted each year")
# plt.xlabel("years")
# plt.ylabel("counts")
# plt.show()
#
# ax = df["project_category"].value_counts().sort_values(ascending=False).plot.bar()
# plt.title("Distribution of categories")
# plt.xlabel("categories")
# plt.ylabel("counts")
# plt.xticks(rotation=25)
# plt.show()
#
# ax = df[df["status"] == "winning"]["project_category"].value_counts().sort_values(ascending=False).plot.bar()
# plt.title("Distribution of categories of winners")
# plt.xlabel("categories")
# plt.ylabel("counts")
# plt.xticks(rotation=25)
# plt.show()

# correlation heatmap
sns_plot = sns.heatmap(df.corr(numeric_only=True), cmap="rocket_r")
plt.xticks(rotation=15)
plt.show()

print(df.info())

df.to_csv(Path(__file__).parent / "data" / "paro_preprocessed.csv")
