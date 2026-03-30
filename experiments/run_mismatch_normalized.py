"""
run_mismatch_normalized.py
Executes the optimal clinical pipeline for Colon Cancer Novelty Detection:
- Mismatch Kernel (m=1) to capture somatic point mutations.
- Normalized Gram Matrix (SVDD equivalence) to handle varying read lengths.
- Ascending Multiple Kernel Learning weights to focus on large structural motifs.

** MODIFIED FOR LOW-RAM TESTING **
"""

import time
import sys
import os
import numpy as np

# Dynamically resolve paths to ensure the script runs from anywhere
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)  # Allow importing from 'src'

from src.data_utils import load_clinical_data
from src.kernels import mixed_string_kernel, normalize_gram
from src.evaluation import evaluate_novelty_detector

def create_mini_fasta(input_path, output_path, max_sequences=100):
    """
    Reads a FASTA file line-by-line and writes a small subset to a new file.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Missing file: {input_path}")
        
    print(f"  -> Creating subset of {os.path.basename(input_path)} ({max_sequences} seqs)...")
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        seq_count = 0
        for line in infile:
            if line.startswith(">"):
                if seq_count >= max_sequences:
                    break
                seq_count += 1
            outfile.write(line)

def generate_mkl_weights(max_k, noise_threshold=2, scaling='linear'):
    """
    Dynamically generates an array of ascending MKL weights.
    
    Parameters:
    - max_k: The maximum k-mer length used in the kernel.
    - noise_threshold: k-mers of this length or smaller get a weight of 0.0.
    - scaling: 'linear' or 'quadratic' growth for the remaining k-mers.
    """
    weights = []
    for k in range(1, max_k + 1):
        if k <= noise_threshold:
            # Silence small k-mers (e.g., 1-mers and 2-mers are just noise)
            weights.append(0.0)
        else:
            # Ascending weight for larger k-mers
            if scaling == 'linear':
                weights.append(float(k - noise_threshold))
            elif scaling == 'quadratic':
                weights.append(float((k - noise_threshold) ** 2))
    
    # Normalize the weights so they sum to exactly 1.0
    total = sum(weights)
    normalized_weights = [round(w / total, 4) for w in weights]
    
    return normalized_weights

def main():
    print("=====================================================")
    print(" COLON CANCER SOMATIC MUTATION DETECTION PIPELINE")
    print("             (LOW-RAM TESTING MODE)")
    print("=====================================================")
    
    # --- 1. Load Clinical Data ---
    data_dir = os.path.join(project_root, 'data')
    normal_file = os.path.join(data_dir, "8kBZ.fa")
    tumor_file = os.path.join(data_dir, "8kBT.fa")
    
    # Define paths for the tiny testing files
    mini_normal_file = os.path.join(data_dir, "8kBZ_mini.fa")
    mini_tumor_file = os.path.join(data_dir, "8kBT_mini.fa")
    
    # Create miniature FASTA files for testing (100 sequences each)
    print("\n[0/4] Preparing low-RAM data subsets...")
    try:
        create_mini_fasta(normal_file, mini_normal_file, max_sequences=2000)
        create_mini_fasta(tumor_file, mini_tumor_file, max_sequences=2000)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
        
    print("\n[1/4] Loading and parsing FASTA files...")
    # PASS THE MINI FILES TO YOUR LOADER INSTEAD OF THE FULL ONES
    train_data, test_data, y_test_true = load_clinical_data(
        normal_fasta=mini_normal_file, 
        tumor_fasta=mini_tumor_file, 
        train_ratio=0.8
    )
    all_data = train_data + test_data
    num_train = len(train_data)
    
    print(f"  -> Successfully loaded {len(all_data)} total sequences for testing.")
    
    # --- 2. Biological Hyperparameters ---
    MAX_K = 6
    MISMATCHES = 1
    NU_PARAM = 0.05  # Expected somatic mutation/noise rate
    
    # Ascending weights: heavily penalizes 1-mers and 2-mers (noise), rewards 5-mers and 6-mers
    mkl_weights = generate_mkl_weights(MAX_K, noise_threshold=2, scaling='linear')
    
    # --- 3. Kernel Computation ---
    print(f"\n[2/4] Computing Mismatch Kernel (k_max={MAX_K}, m={MISMATCHES})...")
    start_time = time.time()
    
    # n_jobs=-1 utilizes all CPU cores for parallel k-mer extraction
    K_full, _ = mixed_string_kernel(
        sequences=all_data, 
        k_max=MAX_K, 
        m=MISMATCHES, 
        weights=mkl_weights,
        n_jobs=-1  
    )
    
    print("[3/4] Normalizing Gram Matrix...")
    K_full = normalize_gram(K_full)
    
    # Slice the precomputed kernel matrix
    K_train = K_full[:num_train, :num_train]
    K_test  = K_full[num_train:, :num_train]
    
    # --- 4. Anomaly Detection ---
    print("\n[4/4] Fitting One-Class SVM and identifying somatic anomalies...")
    metrics = evaluate_novelty_detector(K_train, K_test, y_test_true, nu=NU_PARAM)
    
    elapsed = time.time() - start_time
    
    # --- 5. Output Results ---
    print("\n=====================================================")
    print(" FINAL RESULTS (TESTING SUBSET)")
    print("=====================================================")
    print(f"Kernel Configuration : Mismatch (m={MISMATCHES}), Normalized")
    print(f"Weighting Scheme     : {mkl_weights}")
    print(f"Execution Time       : {elapsed:.2f} seconds")
    print(f"ROC-AUC Score        : {metrics['auc']:.4f}")
    print("\nDetailed Classification Report:")
    print(metrics['report_str'])

    # --- 6. Clean Up Miniature Files ---
    print("\n[Cleaning up temporary test files...]")
    for mini_file in [mini_normal_file, mini_tumor_file]:
        # Remove the miniature FASTA file
        if os.path.exists(mini_file):
            os.remove(mini_file)
        
        # Remove the corresponding FASTA index (.fai) file
        fai_file = mini_file + ".fai"
        if os.path.exists(fai_file):
            os.remove(fai_file)
            
    print("Cleanup complete.")

if __name__ == "__main__":
    main()