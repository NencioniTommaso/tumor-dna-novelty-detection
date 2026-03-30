"""
run_comparisons.py
Main experiment script. Compares Spectrum vs. Mismatch kernels, 
both Normalized and Unnormalized. 
Optimized to run experiments concurrently using ProcessPoolExecutor.
"""

import time
import sys
import os
import concurrent.futures
from typing import Dict, Any, List, Tuple
import numpy as np

# Ensure the 'src' package can be imported if running from the project root or experiments folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_utils import generate_simulated_data, load_clinical_data
from src.kernels import mixed_string_kernel, normalize_gram
from src.evaluation import evaluate_novelty_detector

def run_single_experiment(
    config: Dict[str, Any], 
    all_data: List[str], 
    num_train: int, 
    y_test_true: np.ndarray, 
    max_k: int, 
    mkl_weights: List[float], 
    nu_param: float
) -> Dict[str, Any]:
    """
    Encapsulates a single experimental configuration to be executed in a separate process.
    
    Args:
        config: Dictionary containing the experiment name, mismatch 'm' value, and normalization flag.
        all_data: The full list of DNA sequences (train + test).
        num_train: The integer split point separating training data from test data.
        y_test_true: The ground truth array for the test set.
        max_k: The maximum k-mer length to evaluate.
        mkl_weights: The kernel fusion weights.
        nu_param: The One-Class SVM noise hyperparameter.
        
    Returns:
        A dictionary containing the experiment configuration name, ROC-AUC score, and execution time.
    """
    start_time = time.time()
    
    # 1. Compute Base Kernel
    # NOTE: n_jobs=1 is crucial here! Because we are already parallelizing at the experiment 
    # level using ProcessPoolExecutor, we do not want joblib to spawn nested processes inside 
    # this worker, which would cause CPU context-switching overhead and potential deadlocks.
    K_full, _ = mixed_string_kernel(
        sequences=all_data, 
        k_max=max_k, 
        m=config["m"], 
        weights=mkl_weights,
        n_jobs=1 
    )
    
    # 2. Apply Normalization if configured
    if config["normalize"]:
        K_full = normalize_gram(K_full)
        
    # 3. Slice the Precomputed Kernel
    K_train = K_full[:num_train, :num_train]
    K_test  = K_full[num_train:, :num_train]
    
    # 4. Evaluate the Novelty Detector
    metrics = evaluate_novelty_detector(K_train, K_test, y_test_true, nu=nu_param)
    
    elapsed = time.time() - start_time
    
    return {
        "Config": config["name"],
        "AUC": metrics["auc"],
        "Time (s)": elapsed
    }

def main():
    print("--- 1. Loading Clinical Dataset ---")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    project_root = os.path.dirname(current_dir)

    data_dir = os.path.join(project_root, 'data')
    
    normal_file = os.path.join(data_dir, "8kBZ.fa")
    tumor_file = os.path.join(data_dir, "8kBT.fa")
    
    train_data, test_data, y_test_true = load_clinical_data(
        normal_fasta=normal_file, 
        tumor_fasta=tumor_file, 
        train_ratio=0.8
    )
    
    all_data = train_data + test_data
    num_train = len(train_data)
    
    # Note: Clinical files usually require a lower 'nu' because the healthy 
    # reference genome is highly consistent.
    nu_param = 0.01

    MAX_K = 5
    mkl_weights = [1.0 / MAX_K] * MAX_K  # Equal weights for all
    
    # Define the 4 quadrants of our experiment
    configs = [
        {"name": "Mixed Spectrum (Unnormalized)", "m": 0, "normalize": False},
        {"name": "Mixed Spectrum (Normalized)",   "m": 0, "normalize": True},
        {"name": "Mixed Mismatch (Unnormalized)", "m": 1, "normalize": False},
        {"name": "Mixed Mismatch (Normalized)",   "m": 1, "normalize": True},
    ]
    
    print("\n--- 2. Starting Parallel Kernel Evaluation Matrix ---")
    results = []
    
    # PARALLELIZATION: Run all 4 configurations simultaneously across available CPU cores.
    # The ProcessPoolExecutor creates separate Python memory spaces, allowing us to bypass the GIL.
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit all tasks to the pool
        futures = {
            executor.submit(
                run_single_experiment, 
                config, 
                all_data, 
                num_train, 
                y_test_true, 
                MAX_K, 
                mkl_weights, 
                nu_param
            ): config["name"] for config in configs
        }
        
        # Collect results dynamically as soon as each process finishes
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            results.append(res)
            print(f"-> {res['Config']:<35} completed in {res['Time (s)']:>6.2f}s | ROC-AUC: {res['AUC']:.4f}")

    # 3. Output Final Summary Table
    # Sort results to ensure consistent display order (Spectrum -> Mismatch)
    results.sort(key=lambda x: ("Mismatch" in x["Config"], "Normalized" in x["Config"]))
    
    print("\n\n" + "="*65)
    print(" EXPERIMENT SUMMARY")
    print("="*65)
    print(f"{'Configuration':<35} | {'ROC-AUC':<10} | {'Time (s)':<10}")
    print("-" * 65)
    for res in results:
        print(f"{res['Config']:<35} | {res['AUC']:<10.4f} | {res['Time (s)']:<10.2f}")
    print("="*65)

if __name__ == "__main__":
    main()