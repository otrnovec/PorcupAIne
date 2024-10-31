from settings import DATA_DIR
import os

import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch


def generate_word_embeddings(input):
    """Generate word embeddings and."""

    # Load the Czech BERT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("ufal/robeczech-base")
    model = AutoModel.from_pretrained("ufal/robeczech-base")

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt")

    # Get the embeddings
    with torch.no_grad():
        outputs = model(**inputs)

    # `outputs` has multiple layers; the last hidden state is often used
    # outputs.last_hidden_state is of shape [batch_size, sequence_length, hidden_size]
    embeddings = outputs.last_hidden_state

    # Extract embeddings for each token (each word or subword)
    word_embeddings = embeddings.squeeze().tolist()  # Remove batch dimension if it's 1

    # Extract embeddings for each token (each word or subword)
    # word_embeddings = embeddings.squeeze().tolist()   # Remove batch dimension if it's 1
    # sentence_embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze().tolist()  # Mean pooling, converted to list

    # Calculate mean of all token embeddings for sentence-level embedding
    # sentence_embedding = embeddings.mean(dim=1)

    # # Save embeddings to a .txt file
    # with open("embeddings.txt", "w") as file:
    #     # Save sentence embedding
    #     # file.write("Sentence Embedding:\n")
    #     # file.write(" ".join(map(str, sentence_embedding)) + "\n\n")

    #     # Save word embeddings
    #     file.write("Word Embeddings:\n")
    #     for idx, word_embedding in enumerate(word_embeddings):
    #         file.write(f"Token {idx + 1}: " + " ".join(map(str, word_embedding)) + "\n")
    return word_embeddings

def save_embeddings(embeddings, output_file):
    """Save the embeddings to a .csv file."""
    
    # Save embeddings to a .csv file
    embeddings_df = pd.DataFrame(embeddings)
    print(embeddings_df)
    # columns_to_save = ["project_name","project_description", "public_interest"]
    embeddings_df.to_csv(output_file, index=False)

    print(f"Embeddings saved to {output_file}")


if __name__ == "__main__":
    # Sample text in Czech
    text = "Vytvořit kvalitní české embeddings je náročné."

    # Specify the columns you want to read
    columns_to_read = ["project_name","project_description", "public_interest"]

    input_file = os.path.join(DATA_DIR, "lemmatized_dataset.csv")
    # Read the specified columns from the CSV file
    df = pd.read_csv(input_file, usecols=columns_to_read)
    print(df)

    output_file = os.path.join(DATA_DIR, "embedded_dataset.csv")

    # print(generate_word_embeddings(df))
    # save_embeddings(generate_word_embeddings(df), output_file)
