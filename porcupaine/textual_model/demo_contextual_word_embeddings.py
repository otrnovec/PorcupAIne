import logging
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import StandardScaler


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
        max_length=512  # Adjust as needed
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


def generate_single_instance_embedding(project_name, project_description, public_interest, model_name='ufal/robeczech-base', combine_method='concatenate', weights=None):
    """
    Generate embeddings for a single instance and return the combined embedding.

    Args:
        project_name (str): The project name.
        project_description (str): The project description.
        public_interest (str): The public interest text.
        model_name (str): Name of the pre-trained model.
        combine_method (str): Method to combine embeddings ('concatenate', 'average', 'weighted').
        weights (list of float, optional): Weights for weighted sum. Only used if combine_method is 'weighted'.

    Returns:
        np.ndarray: The combined embedding vector.
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
        return None

    # Prepare the list of texts
    texts = [project_name, project_description, public_interest]
    embeddings = []

    # Process each text input
    for idx, text in enumerate(texts):
        if not text or not isinstance(text, str) or text.strip() == '':
            # Handle missing or empty text by using a zero vector
            embedding = np.zeros(model.config.hidden_size)
            logging.debug(f'Input {idx}: Empty or invalid text encountered.')
        else:
            embedding = get_embedding(text, tokenizer, model)
        embeddings.append(embedding)

    # Combine embeddings
    try:
        combined_embedding = combine_embeddings(
            embeddings,
            method=combine_method,
            weights=weights
        )
    except ValueError as e:
        logging.error(f"Error combining embeddings: {e}")
        return None

    # Optionally normalize the embedding
    scaler = StandardScaler()
    normalized_embedding = scaler.fit_transform(combined_embedding.reshape(1, -1))
    normalized_embedding = normalized_embedding.squeeze()

    return normalized_embedding


if __name__ == '__main__':
    # Example inputs
    project_name = "Example Project Name"
    project_description = "This is an example project description."
    public_interest = "The project aims to benefit the public by providing open access to data."

    # Generate the embedding
    embedding = generate_single_instance_embedding(
        project_name,
        project_description,
        public_interest,
        model_name='ufal/robeczech-base',  # Replace with your desired model
        combine_method='concatenate',       # Options: 'concatenate', 'average', 'weighted'
        weights=None                        # Provide weights if using 'weighted' method
    )

    if embedding is not None:
        print("Combined Embedding:")
        print(embedding)
    else:
        print("Failed to generate embedding.")
