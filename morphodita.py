""" Lemmatize Czech text using the MorphoDiTa library. """

import os
from settings import MORPHODITA_MODEL_DIR

from ufal.morphodita import *


def raw_lemma(lemma_text):
    """Get the raw lemma from the MorphoDiTa output."""
    # in the Morphodita documentation there is a way to get raw lemma but i dont understand it
    lemma_text = lemma_text.split("`")[0]
    lemma_text = lemma_text.split("_")[0]
    lemma_text = lemma_text.split("-")[0]
    return lemma_text


def process_texts(tagger, texts):
    """Process a list of texts and return their lemmas as strings."""
    tokenizer = tagger.newTokenizer()
    if not tokenizer:
        raise Exception("No tokenizer is defined for the supplied model!")

    outputs = []
    for text in texts:
        tokenizer.setText(text)
        lemmas_list = []
        forms = Forms()
        lemmas = TaggedLemmas()
        tokens = TokenRanges()
        while tokenizer.nextSentence(forms, tokens):
            tagger.tag(forms, lemmas)
            # Collect and get raw lemmas
            lemmas_list.extend(raw_lemma(lemma.lemma) for lemma in lemmas)
        outputs.append(" ".join(lemmas_list))
    return outputs


def load_tagger(tagger_file):
    """Load the tagger model from the specified file."""
    tagger = Tagger.load(tagger_file)
    if not tagger:
        raise Exception(f"Cannot load tagger from file '{tagger_file}'")
    return tagger


def get_lemmas(tagger_file, texts):
    """Load the tagger and process the texts to get lemmas."""
    try:
        tagger = load_tagger(tagger_file)
    except Exception as e:
        raise RuntimeError(f"Error loading tagger: {e}") from e

    try:
        outputs = process_texts(tagger, texts)
    except Exception as e:
        raise RuntimeError(f"Error processing texts: {e}") from e

    return outputs


if __name__ == "__main__":
    # Define your tagger file path:
    # download from https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-4794,
    # then place somewhere and then copy the path into @tager_file variable
    # and define your list of texts
    tagger_file = os.path.join(MORPHODITA_MODEL_DIR, "czech-morfflex2.0-pdtc1.0-220710.tagger")
    texts = [
        "Byl jsem včera na návštěvě u babičky. A ta návštěva byla hrozná.",
        "vedlejší příběhy. jeden",
        # Add more text blocks as needed
    ]

    try:
        outputs = get_lemmas(tagger_file, texts)
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)

    # Output the results
    for lemmas_string in outputs:
        print(lemmas_string)
