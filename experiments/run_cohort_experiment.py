"""
run_cohort_experiment.py
Executes a rigorous Patient-Level Machine Learning pipeline for Colon Cancer Novelty Detection.
"""

import time
import sys
import os

# Dynamically resolve paths to ensure the script runs from anywhere
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Import from our custom library and new utils
from src.data_utils import load_patient_cohort
from src.kernels import mixed_string_kernel, normalize_gram
from src.evaluation import evaluate_novelty_detector
from experiments.experiments_utils import setup_logger, parse_arguments, generate_mkl_weights

logger = setup_logger(__name__)

def main():
    args = parse_arguments(project_root)
    
    logger.info("=====================================================")
    logger.info(" COLON CANCER SOMATIC DETECTION: PATIENT COHORT TEST")
    logger.info("=====================================================")
    
    # --- 1. Define the Patient-Level Split ---
    train_normal_files = [
        os.path.join(args.data_dir, f"Healthy_{i}_merged_subset_1200000.fa") for i in range(2, 6)
    ]
    test_normal_files = [
        os.path.join(args.data_dir, f"Healthy_{i}_merged_subset_1200000.fa") for i in range(6, 8)
    ]
    test_tumor_files = [
        os.path.join(args.data_dir, f"Colo_{i}_merged_subset_1200000.fa") for i in range(1, 11) if i != 9 # can be pushed to 38
    ]
    
    # Verify files exist before running
    all_files = train_normal_files + test_normal_files + test_tumor_files
    missing_files = [f for f in all_files if not os.path.exists(f)]
    if missing_files:
        for f in missing_files:
            logger.error(f"Cannot find file: {f}")
        sys.exit(1)

    # --- 2. Load and Sample Data  ---
    logger.info("Starting data loading and sampling...")
    train_data, test_data, y_test_true = load_patient_cohort(
        train_normal_files, 
        test_normal_files, 
        test_tumor_files,
        max_train=args.max_train,
        max_test_normal=args.max_test_normal,
        max_test_tumor=args.max_test_tumor,
        random_seed=args.seed,
        index_cache_dir=args.cache_dir
    )
    
    # --- 3. Kernel Computation ---
    mkl_weights = generate_mkl_weights(args.max_k, args.mismatches)
    logger.info(f"Computing Explicit Sparse Mismatch Kernel (Max K: {args.max_k}, Mismatches: {args.mismatches})...")
    
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
    
    # --- 4. Matrix Slicing & Anomaly Detection ---
    num_train = len(train_data)
    logger.info(f"Fitting One-Class SVM (nu={args.nu_param})...")
    metrics = evaluate_novelty_detector(
        K_train=K_full[:num_train, :num_train], 
        K_test=K_full[num_train:, :num_train], 
        y_test_true=y_test_true, 
        nu=args.nu_param
    )
    
    elapsed = time.time() - start_time
    
    # --- 5. Output Results ---
    logger.info("=====================================================")
    logger.info(" FINAL RESULTS")
    logger.info("=====================================================")
    logger.info(f"Execution Time : {elapsed:.2f} seconds")
    logger.info(f"ROC-AUC Score  : {metrics['auc']:.4f}")
    logger.info(f"\nClassification Report:\n{metrics['report_str']}")

if __name__ == "__main__":
    main()