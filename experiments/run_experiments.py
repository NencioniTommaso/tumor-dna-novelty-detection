"""
run_experiments.py
Main orchestration script for Colon Cancer Novelty Detection.
Compares Spectrum and Mismatch kernels across different biological weighting schemes
using real patient tumor (8kBT.fa) and matched normal (8kBZ.fa) FASTA files.
"""

import time
import sys
import os
import concurrent.futures
from typing import Dict, Any, List
import numpy as np

# 1. Dynamically resolve paths to ensure the script runs from anywhere
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)  # Allow importing from 'src'

from src.data_utils import load_clinical_data
from src.kernels import mixed_string_kernel, normalize_gram
from src.evaluation import evaluate_novelty_detector

def run_single_experiment(
    config: Dict[str, Any], 
    all_data: List[str], 
    num_train: int, 
    y_test_true: np.ndarray, 
    max_k: int, 
    nu_param: float
) -> Dict[str, Any]:
    """
    Executes a single kernel configuration in a dedicated CPU process.
    """
    start_time = time.time()
    
    # 1. Compute Base Kernel (n_jobs=1 prevents nested multiprocessing conflicts)
    K_full, _ = mixed_string_kernel(
        sequences=all_data, 
        k_max=max_k, 
        m=config["m"], 
        weights=config["weights"],
        n_jobs=1 
    )
    
    # 2. Apply Normalization (Mathematically required for varying sequence lengths)
    if config["normalize"]:
        K_full = normalize_gram(K_full)
        
    # 3. Slice the Precomputed Kernel
    K_train = K_full[:num_train, :num_train]
    K_test  = K_full[num_train:, :num_train]
    
    # 4. Evaluate Novelty Detector
    metrics = evaluate_novelty_detector(K_train, K_test, y_test_true, nu=nu_param)
    
    elapsed = time.time() - start_time
    
    return {
        "Config": config["name"],
        "AUC": metrics["auc"],
        "Time (s)": elapsed
    }

def main():
    print("--- 1. Loading Clinical Dataset ---")
    
    # Resolve the data directory
    data_dir = os.path.join(project_root, 'data')
    normal_file = os.path.join(data_dir, "8kBZ.fa")
    tumor_file = os.path.join(data_dir, "8kBT.fa")
    
    # Ensure files exist before starting
    if not os.path.exists(normal_file) or not os.path.exists(tumor_file):
        print(f"ERROR: Could not find FASTA files in {data_dir}")
        print("Please ensure '8kBZ.fa' and '8kBT.fa' are placed in the 'data' folder.")
        sys.exit(1)
        
    # Load and split the data (80% of normal data used to build the baseline)
    train_data, test_data, y_test_true = load_clinical_data(
        normal_fasta=normal_file, 
        tumor_fasta=tumor_file, 
        train_ratio=0.8
    )
    
    all_data = train_data + test_data
    num_train = len(train_data)
    
    # --- Biological Hyperparameters ---
    MAX_K = 6
    nu_param = 0.01  # Lower noise bound for highly conserved clinical genomes
    
    # Define different biological hypotheses for k-mer importance
    weight_schemes = {
        "Uniform":   [0.16, 0.16, 0.16, 0.16, 0.16, 0.16],
        "Ascending": [0.00, 0.05, 0.10, 0.20, 0.30, 0.35], # Penalizes short noise
        "Strict":    [0.00, 0.00, 0.00, 0.20, 0.35, 0.45]  # Only looks at 4, 5, 6-mers
    }
    
    configs = []
    # We will test both Spectrum (Exact) and Mismatch (Mutated) to prove the hypothesis
    for scheme_name, weights in weight_schemes.items():
        configs.append({
            "name": f"Spectrum ({scheme_name})", 
            "m": 0, 
            "normalize": True, 
            "weights": weights
        })
        configs.append({
            "name": f"Mismatch ({scheme_name})", 
            "m": 1, 
            "normalize": True, 
            "weights": weights
        })

    print(f"\n--- 2. Starting Parallel Kernel Evaluation ({len(configs)} Configs) ---")
    results = []
    
    # Execute all configurations concurrently across available CPU cores
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                run_single_experiment, 
                config, 
                all_data, 
                num_train, 
                y_test_true, 
                MAX_K, 
                nu_param
            ): config["name"] for config in configs
        }
        
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            results.append(res)
            print(f"-> {res['Config']:<25} completed in {res['Time (s)']:>6.2f}s | ROC-AUC: {res['AUC']:.4f}")

    # --- 3. Output Final Scientific Summary Table ---
    # Sort results first by kernel type (Spectrum vs Mismatch), then by AUC performance
    results.sort(key=lambda x: ("Mismatch" in x["Config"], -x["AUC"]))
    
    print("\n\n" + "="*65)
    print(" CLINICAL COLON CANCER EXPERIMENT SUMMARY")
    print("="*65)
    print(f"{'Configuration':<28} | {'ROC-AUC':<10} | {'Time (s)':<10}")
    print("-" * 65)
    for res in results:
        print(f"{res['Config']:<28} | {res['AUC']:<10.4f} | {res['Time (s)']:<10.2f}")
    print("="*65)

if __name__ == "__main__":
    main()