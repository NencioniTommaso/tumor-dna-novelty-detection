"""
run_cohort_experiment.py
Executes a rigorous Patient-Level Machine Learning pipeline for Colon Cancer Novelty Detection.
Includes random sampling to bound memory execution to 10,000 total sequences.
"""

import time
import sys
import os
import numpy as np

# Dynamically resolve paths to ensure the script runs from anywhere
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Import from our custom library
from src.data_utils import load_patient_cohort
from src.kernels import mixed_string_kernel, normalize_gram
from src.evaluation import evaluate_novelty_detector

def generate_mkl_weights(max_k: int, noise_threshold: int = 2, scaling: str = 'linear') -> list[float]:
    """
    Dynamically generates an array of ascending Multiple Kernel Learning (MKL) weights.
    Silences small k-mers (noise) and rewards larger structural motifs.
    """
    weights = []
    for k in range(1, max_k + 1):
        if k <= noise_threshold:
            weights.append(0.0)
        else:
            if scaling == 'linear':
                weights.append(float(k - noise_threshold))
            elif scaling == 'quadratic':
                weights.append(float((k - noise_threshold) ** 2))
                
    total = sum(weights)
    return [round(w / total, 4) for w in weights]


def main():
    print("=====================================================")
    print(" COLON CANCER SOMATIC DETECTION: PATIENT COHORT TEST")
    print("=====================================================")
    
    data_dir = os.path.join(project_root, 'data')
    
    # --- 1. Define the Patient-Level Split ---
    train_normal_files = [
        os.path.join(data_dir, "Healthy_2_merged_subset_1200000.fa"),
        os.path.join(data_dir, "Healthy_3_merged_subset_1200000.fa"),
        os.path.join(data_dir, "Healthy_4_merged_subset_1200000.fa"),
        os.path.join(data_dir, "Healthy_5_merged_subset_1200000.fa")
    ]
    
    test_normal_files = [
        os.path.join(data_dir, "Healthy_6_merged_subset_1200000.fa"),
        os.path.join(data_dir, "Healthy_7_merged_subset_1200000.fa")
    ]
    
    test_tumor_files = [
        os.path.join(data_dir, "Colo_11_merged_subset_1200000.fa"),
        os.path.join(data_dir, "Colo_12_merged_subset_1200000.fa"),
        os.path.join(data_dir, "Colo_13_merged_subset_1200000.fa")
    ]
    
    # Verify files exist before running
    all_files = train_normal_files + test_normal_files + test_tumor_files
    for f in all_files:
        if not os.path.exists(f):
            print(f"CRITICAL ERROR: Cannot find file {f}")
            sys.exit(1)

    # --- 2. Load and Sample Data  ---
    # Train Normal + Test Normal + Test Tumor
    train_data, test_data, y_test_true = load_patient_cohort(
        train_normal_files, 
        test_normal_files, 
        test_tumor_files,
        max_train=5000,
        max_test_normal=2500,
        max_test_tumor=2500,
        random_seed=42  # Ensures reproducibility across experiment runs
    )
    
    all_data = train_data + test_data
    num_train = len(train_data)
    
    # --- 3. Biological Hyperparameters ---
    MAX_K = 6
    MISMATCHES = 1
    NU_PARAM = 0.2  # The expected anomaly rate
    
    mkl_weights = generate_mkl_weights(MAX_K, noise_threshold=2, scaling='linear')
    
    print(f"\n--- Kernel Configuration ---")
    print(f"K-max      : {MAX_K}")
    print(f"Mismatches : {MISMATCHES}")
    print(f"MKL Weights: {mkl_weights}")
    
    # --- 4. Kernel Computation ---
    print(f"\n[Computing Explicit Sparse Mismatch Kernel...]")
    start_time = time.time()
    
    K_full, _ = mixed_string_kernel(
        sequences=all_data, 
        k_max=MAX_K, 
        m=MISMATCHES, 
        weights=mkl_weights,
        n_jobs=-1  
    )
    
    print("[Normalizing Gram Matrix...]")
    K_full = normalize_gram(K_full)
    
    # --- 5. Matrix Slicing ---
    K_train = K_full[:num_train, :num_train]
    K_test  = K_full[num_train:, :num_train]
    
    # --- 6. Anomaly Detection (One-Class SVM) ---
    print("\n[Fitting One-Class SVM...]")
    metrics = evaluate_novelty_detector(
        K_train=K_train, 
        K_test=K_test, 
        y_test_true=y_test_true, 
        nu=NU_PARAM
    )
    
    elapsed = time.time() - start_time
    
    # --- 7. Output Results ---
    print("\n=====================================================")
    print(" FINAL RESULTS: PATIENT COHORT VALIDATION")
    print("=====================================================")
    print(f"Execution Time       : {elapsed:.2f} seconds")
    print(f"ROC-AUC Score        : {metrics['auc']:.4f}")
    print("\nDetailed Classification Report:")
    print(metrics['report_str'])
    print("=====================================================")

if __name__ == "__main__":
    main()