from settings import DATA_DIR, MORPHODITA_MODEL_DIR
import os

import pandas as pd
from morphodita import get_lemmas


def generate_lemmatized_dataset(dataframe: pd.DataFrame, tagger_file, output_file):
    """Generate lemmatized text data from the dataframe."""

    name_lemmatized = get_lemmas(tagger_file, dataframe["project_name"].tolist())
    description_lemmatized = get_lemmas(
        tagger_file, dataframe["project_description"].tolist()
    )
    interest_lemmatized = get_lemmas(tagger_file, dataframe["public_interest"].tolist())
    df = pd.DataFrame(
        {
            "project_name": name_lemmatized,
            "project_description": description_lemmatized,
            "public_interest": interest_lemmatized,
            "status": dataframe["status"],
        }
    )

    df["status"].replace(
        {
            "feasible": 1,
            "winning": 1,
            "unfeasible": 1,
            "proveditelný": 1,
            "without support": 0,
        },
        inplace=True,
    )

    df.to_csv(DATA_DIR / output_file, index=False)


if __name__ == "__main__":
    # Replace with your tagger file path
    tagger_file = os.path.join(
        MORPHODITA_MODEL_DIR, "czech-morfflex2.0-pdtc1.0-220710.tagger"
    )
    df = pd.read_csv(DATA_DIR / "paro_preprocessed.csv")

    generate_lemmatized_dataset(df, tagger_file, "lemmatized_dataset.csv")
