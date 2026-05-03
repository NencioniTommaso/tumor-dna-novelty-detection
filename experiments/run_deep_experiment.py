"""
run_deep_experiment.py
Executes a Deep Learning-Based Kernel ML pipeline for Colon Cancer Novelty Detection.
"""

import time
import sys
import os
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, roc_auc_score

# Dynamically resolve paths to ensure the script runs from anywhere
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Import from our custom library and experiment utils
from src.data_utils import load_patient_cohort
from src.DNAFeatureExtractor import compute_train_test_kernels
# We can remove normalize_gram entirely, as the RBF kernel doesn't need it
from experiments.experiments_utils import setup_logger, parse_arguments

logger = setup_logger(__name__)

def main():
    args = parse_arguments(project_root)
    
    logger.info("=====================================================")
    logger.info(" COLON CANCER SOMATIC DETECTION: DEEP LEARNING KERNEL")
    logger.info("=====================================================")
    
    # --- 1. Define the Patient-Level Split ---
    train_normal_files = [
        os.path.join(args.data_dir, f"Healthy_{i}_merged_subset_1200000.fa") for i in range(2, 6)
    ]
    test_normal_files = [
        os.path.join(args.data_dir, f"Healthy_{i}_merged_subset_1200000.fa") for i in range(6, 8)
    ]
    test_tumor_files = [
        os.path.join(args.data_dir, f"Colo_{i}_merged_subset_1200000.fa") for i in range(11, 14)
    ]
    
    # Verify files exist before running
    all_files = train_normal_files + test_normal_files + test_tumor_files
    missing_files = [f for f in all_files if not os.path.exists(f)]
    if missing_files:
        for f in missing_files:
            logger.error(f"Cannot find file: {f}")
        sys.exit(1)

    # --- 2. Load and Sample Data ---
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
    logger.info("Computing Deep Learning Kernels (DNABERT-2)...")
    
    start_time = time.time()
    
    # K_train is (Train vs Train), K_test is (Test vs Train)
    K_train, K_test = compute_train_test_kernels(
        train_sequences=train_data,
        test_sequences=test_data,
        model_name="quietflamingo/dnabert2-no-flashattention",  # <--- The CPU-friendly clone
        kernel_type="rbf",
        batch_size=8            #arcgs.batch_size is set to 8 by default to ensure it runs on CPU without OOM
    )
    
    # Normalization isn't needed for RBF

    # --- 4. Matrix Slicing & Anomaly Detection ---
    logger.info(f"Fitting One-Class SVM (nu={args.nu_param})...")
    
    svm = OneClassSVM(kernel='precomputed', nu=args.nu_param)
    svm.fit(K_train)
    
    predictions = svm.predict(K_test)
    anomaly_scores = svm.decision_function(K_test)
    
    # Invert scores for ROC-AUC (negative scores indicate anomalies)
    auc = roc_auc_score(y_test_true == -1, -anomaly_scores)
    report_str = classification_report(
        y_test_true,
        predictions,
        target_names=['Cancer (-1)', 'Healthy (1)'],
        zero_division=0
    )
    
    elapsed = time.time() - start_time
    
    # --- 5. Output Results ---
    logger.info("=====================================================")
    logger.info(" FINAL RESULTS")
    logger.info("=====================================================")
    logger.info(f"Execution Time : {elapsed:.2f} seconds")
    logger.info(f"ROC-AUC Score  : {auc:.4f}")
    logger.info(f"\nClassification Report:\n{report_str}")

if __name__ == "__main__":
    main()