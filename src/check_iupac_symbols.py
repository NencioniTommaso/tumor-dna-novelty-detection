import argparse
import os
import sys

def check_iupac_presence(fasta_path):
    """
    Checks for the existence of IUPAC nucleotide symbols in a FASTA file.
    Optimized for large datasets using lazy loading and early stopping.
    """
    # Standard IUPAC nucleotide alphabet (including gap '-')
    iupac_alphabet = set("ACGTURYKMSWBDHVN-")
    found_symbols = set()
    
    print(f"Scanning '{fasta_path}' for vocabulary...")

    # Sanity check if the file exists before doing anything
    if not os.path.exists(fasta_path):
        print(f"❌ Error: Could not find file at '{fasta_path}'")
        sys.exit(1)

    try:
        with open(fasta_path, 'r') as file:
            for line in file:
                # Skip FASTA sequence header lines
                if line.startswith('>'):
                    continue
                
                # Strip whitespace/newlines, uppercase, and add to the set
                found_symbols.update(line.strip().upper())
                
                # EARLY STOPPING: 
                # If our found symbols completely cover the IUPAC alphabet, 
                # there is no need to read the rest of the file.
                if iupac_alphabet.issubset(found_symbols):
                    print("✅ All IUPAC symbols found! Stopping scan early.")
                    break
                    
    except Exception as e:
        print(f"❌ An error occurred while reading the file: {e}")
        sys.exit(1)

    # --- Print the Results ---
    print("\n--- IUPAC Symbol Presence ---")
    for symbol in sorted(iupac_alphabet):
        status = "✅ Present" if symbol in found_symbols else "❌ Missing"
        print(f"{symbol}: {status}")

    # --- ML Data Sanity Check ---
    unexpected_chars = found_symbols - iupac_alphabet
    if unexpected_chars:
        print(f"\n⚠️ DATA WARNING: Found unexpected non-IUPAC characters: {unexpected_chars}")
        print("You may need to clean your dataset or add an <UNK> token to your ML tokenizer.")
    else:
        print("\n✅ DATA CLEAN: No unexpected characters found outside the standard IUPAC alphabet.")

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description="A fast CLI tool to scan a FASTA file and check which IUPAC nucleotide symbols are present."
    )
    
    # Define the required file path argument
    parser.add_argument(
        "fasta_file", 
        type=str, 
        help="Path to the .fa or .fasta file to be scanned."
    )
    
    # Parse the arguments from the terminal
    args = parser.parse_args()
    
    # Execute the logic
    check_iupac_presence(args.fasta_file)

if __name__ == "__main__":
    main()