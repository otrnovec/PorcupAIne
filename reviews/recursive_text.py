import argparse
import math


# Recursive function to remove the last letter from the last word for every word in text
def remove_last_letter(text):
    if not text.strip():                                  return text
    words = text.split()
    words[-1] = words[-1][:-1]
    return remove_last_letter(' '.join(words[:-1])) + ' ' + words[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recursively remove the last letter from the last word.")
    parser.add_argument('text', help='The text to process.')
    args = parser.parse_args()

    # Call the recursive function
    print(remove_last_letter(args.text))
