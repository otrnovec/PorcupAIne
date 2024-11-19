""" After getting suspicion that our original word embeddings (first_word_embeddings.py) 
have issues, I have decided to try both contextual (generally DONT require lemmatization and removal of stop-words) 
and non-contextual embeddings (Generally require lemmatization and removal of stop-words)."""

import logging
import os
from settings import DATA_DIR


import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer


def get_embedding(text, tokenizer, model):
    """
    Generate the embedding for a given text using the specified tokenizer and model.

    Args:
        text (str): The input text.
        tokenizer: The tokenizer instance.
        model: The model instance.

    Returns:
        np.ndarray: The embedding vector as a NumPy array.
    """
    # Tokenize the input text
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=512  # Set max_length to handle very long texts
    )
    # Get the model outputs without tracking gradients
    with torch.no_grad():
        outputs = model(**inputs)
    # Extract the embeddings from the [CLS] token
    embedding = outputs.last_hidden_state[:, 0, :]
    return embedding.squeeze().numpy()


def combine_embeddings(embeddings, method='concatenate', weights=None):
    """
    Combine a list of embeddings into a single embedding vector.

    Args:
        embeddings (list of np.ndarray): List of embedding vectors.
        method (str): Method to combine embeddings ('concatenate', 'average', 'weighted').
        weights (list of float, optional): Weights for weighted sum. Only used if method is 'weighted'.

    Returns:
        np.ndarray: The combined embedding vector.
    """
    if method == 'concatenate':
        combined_embedding = np.concatenate(embeddings)
    elif method == 'average':
        combined_embedding = np.mean(embeddings, axis=0)
    elif method == 'weighted':
        if weights is None or len(weights) != len(embeddings):
            raise ValueError("Weights must be provided and match the number of embeddings for weighted combination.")
        combined_embedding = np.average(embeddings, axis=0, weights=weights)
    else:
        raise ValueError(f"Invalid combination method: {method}")
    return combined_embedding


def main(input_file, output_file, columns=None, model_name='ufal/robeczech-base', combine_method='concatenate', weights=None ):
    """
    Main function to read input CSV, generate embeddings, and save to output CSV.

    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to the output CSV file.
        columns (list of str, optional): List of columns to process. If None, all columns are used.
        model_name (str): Name of the pre-trained model.
        combine_method (str): Method to combine embeddings ('concatenate', 'average', 'weighted').
        weights (list of float, optional): Weights for weighted sum. Only used if combine_method is 'weighted'.
        
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info('Loading tokenizer and model...')

    # Load the tokenizer and model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    except Exception as e:
        logging.error(f"Error loading model '{model_name}': {e}")
        return

    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        logging.error(f"Error reading input CSV file '{input_file}': {e}")
        return

    # Filter the DataFrame to use only the specified columns if provided
    if columns:
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            logging.error(f"Columns not found in the input CSV: {missing_columns}")
            return
        df = df[columns]

    logging.info(f'Processing {len(df)} rows for columns: {columns if columns else df.columns.tolist()}')
    
    combined_embeddings = []

    # Process each row in the DataFrame
    for index, row in df.iterrows():
        embeddings = []
        for col in df.columns:
            text = row[col]
            if pd.isnull(text) or not isinstance(text, str) or text.strip() == '':
                # Handle missing or empty text by using a zero vector
                embedding = np.zeros(model.config.hidden_size)
                logging.debug(f'Row {index}, column "{col}": Empty or invalid text encountered.')
            else:
                embedding = get_embedding(text, tokenizer, model)
            embeddings.append(embedding)
        
        try:
            combined_embedding = combine_embeddings(
                embeddings,
                method=combine_method,
                weights=weights
            )
        except ValueError as e:
            logging.error(f"Error combining embeddings at row {index}: {e}")
            return
        combined_embeddings.append(combined_embedding)

    combined_embeddings = np.array(combined_embeddings)

    scaler = StandardScaler()
    normalized_embeddings = scaler.fit_transform(combined_embeddings)

    embeddings_df = pd.DataFrame(normalized_embeddings)

    try:
        embeddings_df.to_csv(output_file, index=False)
        logging.info(f"Embeddings saved to {output_file}")
    except Exception as e:
        logging.error(f"Error saving embeddings to CSV file '{output_file}': {e}")
        return

def add_status_to_embedded_dataset(embedded_dataset_file, lemmatized_dataset_file):
    """Add the 'status' column to the embedded dataset.
    Args:
        embedded_dataset_file (csv): path to the embedded dataset file,
        lemmatized_dataset_file (csv): path to a dataset with binary status column.
        
    """

    embedded_df = pd.read_csv(embedded_dataset_file)

    lemmatized_df = pd.read_csv(lemmatized_dataset_file)

    embedded_df["status"] = lemmatized_df["status"]

    embedded_df.to_csv(embedded_dataset_file, index=False)
    print(f"Status added to {embedded_dataset_file}")



if __name__ == '__main__':
    # Specify your parameters here
    input_file = os.path.join(DATA_DIR, "paro_preprocessed.csv")
    output_file = os.path.join(DATA_DIR, "contextual_embeddings.csv")
    model_name = 'ufal/robeczech-base'  
    combine_method = 'concatenate'  
    weights = None  # Weights for weighted combination, e.g., [0.2, 0.4, 0.4]
    lemma_no_stop_words_file = os.path.join(
        DATA_DIR, "lemmatized_no_stopwords_dataset.csv"
    )
    columns_to_read = [
        "project_name",
        "project_description",
        "public_interest"]

    main(input_file, output_file, columns_to_read, model_name, combine_method, weights)
    add_status_to_embedded_dataset(output_file, lemma_no_stop_words_file)
