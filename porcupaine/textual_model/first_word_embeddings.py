import os

import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch

from porcupaine.settings import *


# Load the Czech BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("ufal/robeczech-base")
model = AutoModel.from_pretrained("ufal/robeczech-base")


def generate_word_embeddings(text):
    """Generate word embeddings for a given text, with truncation if too long."""
    # Tokenize with truncation for long texts
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

    # Get the embeddings
    with torch.no_grad():
        outputs = model(**inputs)

    # Calculate mean of all token embeddings for a sentence-level embedding
    sentence_embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze().tolist()

    return sentence_embedding


def process_and_save_embeddings(input_file, output_file, columns_to_read):
    """Read specified columns from a CSV, generate embeddings, and save to a new CSV."""

    # Read the specified columns from the CSV file
    df = pd.read_csv(input_file, usecols=columns_to_read)

    # Initialize an output DataFrame
    output_df = pd.DataFrame()

    # Generate embeddings for each column and add to output DataFrame
    for column in columns_to_read:
        output_df[column] = df[column].apply(lambda x: generate_word_embeddings(str(x)))

    # Save to CSV
    output_df.to_csv(output_file, index=False)
    print(f"Embeddings saved to {output_file}")


def add_status_to_embedded_dataset(embedded_dataset_file, lemmatized_dataset_file):
    """Add the 'status' column to the embedded dataset."""

    # Read the embedded dataset
    embedded_df = pd.read_csv(embedded_dataset_file)

    # Read the lemmatized dataset
    lemmatized_df = pd.read_csv(lemmatized_dataset_file)

    # Add the 'status' column to the embedded dataset
    embedded_df["status"] = lemmatized_df["status"]

    # Save the updated dataset
    embedded_df.to_csv(embedded_dataset_file, index=False)
    print(f"Status added to {embedded_dataset_file}")


def remove_stop_words_from_csv(input_csv, output_csv, stopwords_txt, text_columns):
    """
    Remove stop words from specified columns in a CSV file and save the cleaned data to a new CSV file.

    Parameters:
    input_csv (str): Path to the input CSV file.
    output_csv (str): Path to save the output CSV file.
    stopwords_txt (str): Path to the text file containing stop words (one per line).
    text_columns (list[str]): List of column names to process.
    """
    # Load stop words from the text file and normalize them
    with open(stopwords_txt, "r", encoding="utf-8") as f:
        stop_words = set(word.strip().lower() for word in f.read().splitlines())

    # Load the input CSV file
    df = pd.read_csv(input_csv)

    # Remove stop words from specified columns
    for column in text_columns:
        if column in df.columns:
            df[column] = (
                df[column]
                .astype(str)
                .apply(
                    lambda x: " ".join(
                        [
                            word
                            for word in x.split()
                            if word.lower().strip() not in stop_words
                        ]
                    )
                )
            )

    # Save the cleaned DataFrame to the output CSV file
    df.to_csv(output_csv, index=False)
    print(f"Stop words removed and data saved to {output_csv}")


if __name__ == "__main__":
    lemma_file = os.path.join(DATA_DIR, "lemmatized_dataset.csv")
    lemma_no_stop_words_file = os.path.join(
        DATA_DIR, "lemmatized_no_stopwords_dataset.csv"
    )
    columns_to_read = [
        "project_name",
        "project_description",
        "public_interest",
        "status",
    ]
    remove_stop_words_from_csv(
        lemma_file, lemma_no_stop_words_file, "czech_stopwords.txt", columns_to_read
    )

    embeddings_file = os.path.join(DATA_DIR, "embedded_dataset.csv")

    columns_to_read_emb = ["project_name", "project_description", "public_interest"]
    # Process and save embeddings
    process_and_save_embeddings(
        lemma_no_stop_words_file, embeddings_file, columns_to_read_emb
    )
    add_status_to_embedded_dataset(embeddings_file, lemma_no_stop_words_file)
