from settings import DATA_DIR
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch

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
        output_df[column + "_embedding"] = df[column].apply(lambda x: generate_word_embeddings(str(x)))
    
    # Save to CSV
    output_df.to_csv(output_file, index=False)
    print(f"Embeddings saved to {output_file}")

if __name__ == "__main__":
    # Specify input and output files
    input_file = os.path.join(DATA_DIR, "lemmatized_dataset.csv")
    output_file = os.path.join(DATA_DIR, "embedded_dataset.csv")
    columns_to_read = ["project_name", "project_description", "public_interest"]

    # Process and save embeddings
    process_and_save_embeddings(input_file, output_file, columns_to_read)
