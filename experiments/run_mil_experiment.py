"""
run_mil_experiment.py
Executes a Patient-Level Multiple Instance Learning (MIL) pipeline for Colon Cancer Novelty Detection.
Aggregates sequence-level anomaly scores to evaluate true patient-level ROC-AUC.
"""

import time
import sys
import os
import numpy as np
from sklearn.metrics import roc_auc_score

# Dynamically resolve paths to ensure the script runs from anywhere
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Import from our custom library
from src.data_utils import load_tracked_patient_cohort
from src.kernels import generate_mkl_weights, mixed_string_kernel, normalize_gram
from src.evaluation import evaluate_novelty_detector, evaluate_patient_level_novelty
from experiments.experiments_utils import setup_logger, parse_arguments

logger = setup_logger(__name__)


def main():
    args = parse_arguments(project_root)
    
    logger.info("=====================================================")
    logger.info(" COLON CANCER SOMATIC DETECTION: MULTIPLE INSTANCE LEARNING")
    logger.info("=====================================================")
    
    # --- 1. Define the Patient-Level Split ---
    train_normal_files = [
        os.path.join(args.data_dir, f"Healthy_{i}_merged_subset_1200000.fa") for i in range(2, 6)
    ]
    test_normal_files = [
        os.path.join(args.data_dir, f"Healthy_{i}_merged_subset_1200000.fa") for i in range(6, 8)
    ]
    test_tumor_files = [
        os.path.join(args.data_dir, f"Colo_{i}_merged_subset_1200000.fa") for i in range(1, 11) if i != 9
    ]
    
    # Verify files exist before running
    all_files = train_normal_files + test_normal_files + test_tumor_files
    missing_files = [f for f in all_files if not os.path.exists(f)]
    if missing_files:
        for f in missing_files:
            logger.error(f"Cannot find file: {f}")
        sys.exit(1)

    # --- 2. Load and Sample Tracked Data  ---
    logger.info("\nStarting data loading and tracking...")
    train_data, test_data, y_test_true_seq, test_files_info = load_tracked_patient_cohort(
        train_normal_files, 
        test_normal_files, 
        test_tumor_files,
        args,
        logger
    )
    
    # --- 3. Kernel Computation ---
    mkl_weights = generate_mkl_weights(args.max_k, noise_threshold=max(1, 2 * args.mismatches))
    logger.info(f"\nComputing Explicit Sparse Mismatch Kernel (Max K: {args.max_k}, Mismatches: {args.mismatches})...")
    
    start_time = time.time()
    K_full, _ = mixed_string_kernel(
        sequences=train_data + test_data, 
        k_max=args.max_k, 
        m=args.mismatches, 
        weights=mkl_weights,
        n_jobs=args.n_jobs  
    )
    
    logger.info("Normalizing Gram Matrix...")
    K_full = normalize_gram(K_full)
    
    # --- 4. Matrix Slicing & Sequence Anomaly Detection ---
    num_train = len(train_data)
    logger.info(f"\nFitting One-Class SVM (nu={args.nu_param})...")
    
    # Sequence-level predictions (This is noisy, as healthy reads in a tumor patient are labeled -1)
    metrics = evaluate_novelty_detector(
        K_train=K_full[:num_train, :num_train], 
        K_test=K_full[num_train:, :num_train], 
        y_test_true=y_test_true_seq, 
        nu=args.nu_param
    )
    
    # --- 5. True Patient-Level Anomaly Aggregation ---
    patient_auc = evaluate_patient_level_novelty(metrics['anomaly_scores'], test_files_info, logger)
    
    elapsed = time.time() - start_time
    
    # --- 6. Output Results ---
    logger.info("\n=====================================================")
    logger.info(" FINAL RESULTS: PATIENT-LEVEL MIL EVALUATION")
    logger.info("=====================================================")
    logger.info(f"Execution Time          : {elapsed:.2f} seconds")
    logger.info(f"Sequence-Level ROC-AUC  : {metrics['auc']:.4f} (Expected to be low/noisy)")
    logger.info(f"PATIENT-LEVEL ROC-AUC   : {patient_auc:.4f} (True Clinical Metric)")
    logger.info("=====================================================")

if __name__ == "__main__":
    main()