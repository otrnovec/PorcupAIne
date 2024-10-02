import argparse
import random
import re

def main(text):
    """Change order of words in a sentence. Do not change words themselves. No splitting/merging."""
    chunks = [chunk for chunk in re.split(r'\b', text)]
    random.shuffle(chunks)
    return ''.join(chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load text from command line argument and randomize it.")
    parser.add_argument('--text', help='Any text.')
    args = parser.parse_args()
    result   = main(args.text)
    print(f"Received text: {result}")