# -*- coding: utf-8 -*-
import argparse  # Import the arsenal
import sys  # We'll need this to manipulate input/output streams

def reverse_da_string(z):  # Function that reverses the text; we don't play nice here.
    return z[::-1]  # Simple, effective, dangerous

if __name__ == "__main__":  # This is where the magic happens. You call, we deliver.
    parser = argparse.ArgumentParser(description="Feed me your text, and I'll hack it...")
    parser.add_argument('txt', type=str, help="Give me your text")
    args = parser.parse_args()  # Lock and load
    # Engage with the target
    if args.txt:
        hacked_text = reverse_da_string(args.txt)
        print(f"\n>>> [ REVERSED DATA OUTPUT ] <<<")
        print(hacked_text)
    else:
        # You didn't give me anything, rookie mistake.
        sys.stderr.write("Error: No input text provided. Try harder...\n")
        sys.exit(1)