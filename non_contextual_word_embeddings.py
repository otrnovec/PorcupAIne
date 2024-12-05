""" generate non-contextual embedings, requires lemmatization and removal of stop-words"""

import os

import re 
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors

from settings import DATA_DIR

def load_gensim_fasttext_model(model_path):
    """
    Load the pre-trained fastText model using gensim.
    Args:
        model_path (str): Path to the pre-trained .vec file.
    Returns:
        model: gensim KeyedVectors model.
    """
    print("Loading pre-trained fastText model using gensim...")
    model = KeyedVectors.load_word2vec_format(model_path, binary=False)
    return model

def preprocess_text(text):
    """
    Additional preprocessing on the text.
    - Lowercases text
    - Removes extra spaces
    Args:
        text (str): Input text to preprocess.
    Returns:
        str: Cleaned text.
    """
    text = text.lower()
    return re.sub("\s{2,}", " ", text).strip().lower()  

def text_to_embedding(text, model):
    """
    Convert a preprocessed text into an averaged fastText embedding using gensim.
    Args:
        text (str): Preprocessed input text.
        model: gensim KeyedVectors model.
    Returns:
        np.ndarray: Averaged embedding vector for the text.
    """
    words = text.split()
    embeddings = [model[word] for word in words if word in model]
    
    if not embeddings:
        return np.zeros(model.vector_size)  # Return a zero vector if no words are found in the model
    return np.mean(embeddings, axis=0)
def process_csv(input_file, output_file, model_path, *columns):
    """
    Read the CSV, convert text to embeddings, and save to a new CSV.
    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the output CSV file.
        model_path (str): Path to the pre-trained fastText model.
        *columns (str): columns used.
    """
    # Load gensim fastText model
    model = load_gensim_fasttext_model(model_path)

    # Read input CSV
    df = pd.read_csv(input_file)
    
    combined_embeddings = []

    # Process each row in the DataFrame
    for _, row in df.iterrows():
        embeddings = []
        for col in columns:
            text = row[col]
            if pd.isnull(text) or not isinstance(text, str) or text.strip() == '':
                embedding = np.zeros(model.vector_size)
            else:
                text = preprocess_text(text)
                embedding = text_to_embedding(text, model)
            embeddings.append(embedding)
        # Concatenate embeddings from the three columns
        combined_embedding = np.concatenate(embeddings)
        combined_embeddings.append(combined_embedding)

    # Convert to DataFrame and save to CSV
    embeddings_df = pd.DataFrame(combined_embeddings)
    embeddings_df.to_csv(output_file, index=False)
    print(f"Embeddings saved to '{output_file}'")

def add_status_to_embedded_dataset(output_file, input_file):
    """
    Adds the status column from the original dataset to the embedded dataset.
    """
    df_embeddings = pd.read_csv(output_file)
    df_original = pd.read_csv(input_file)

    # Assuming the status column exists in the original dataset
    df_embeddings['status'] = df_original['status']
    df_embeddings.to_csv(output_file, index=False)
    print(f"Status column added to '{output_file}'")

if __name__ == '__main__':
    model_path = "fasttext_model/cc.cs.300.vec"
    input_file = os.path.join(DATA_DIR, "lemmatized_no_stopwords_dataset.csv")
    output_file = os.path.join(DATA_DIR, "non_contextual_embeddings.csv")

    # Specify the column names that contain the text
    col1 = 'project_name'
    col2 = 'project_description'
    col3 = 'public_interest'

    process_csv(input_file, output_file, model_path, col1, col2, col3)

    add_status_to_embedded_dataset(output_file, input_file)
