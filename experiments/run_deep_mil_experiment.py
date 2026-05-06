"""
run_deep_mil_experiment.py
Executes a Deep Learning-Based Patient-Level Multiple Instance Learning (MIL)
pipeline for Colon Cancer Novelty Detection.
"""

import time
import sys
import os

# Dynamically resolve paths to ensure the script runs from anywhere
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Import from our custom library
from src.data_utils import load_tracked_patient_cohort
from src.DNAFeatureExtractor import compute_train_test_kernels
from src.evaluation import evaluate_novelty_detector, evaluate_patient_level_novelty
from experiments.experiments_utils import setup_logger, parse_arguments

logger = setup_logger(__name__)


def main():
    args = parse_arguments(project_root)
    
    logger.info("=====================================================")
    logger.info(" COLON CANCER SOMATIC DETECTION: DEEP LEARNING MIL")
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

    # --- 2. Load and Sample Tracked Data ---
    logger.info("Starting data loading and tracking...")
    train_data, test_data, y_test_true_seq, test_files_info = load_tracked_patient_cohort(
        train_normal_files,
        test_normal_files,
        test_tumor_files,
        args,
        logger
    )
    
    # --- 3. Deep Kernel Computation ---
    start_time = time.time()
    
    # K_train is (Train vs Train), K_test is (Test vs Train)
    K_train, K_test = compute_train_test_kernels(
        train_sequences=train_data,
        test_sequences=test_data,
        model_name="quietflamingo/dnabert2-no-flashattention", 
        kernel_type="rbf",
        batch_size=8
    )
    
    # --- 4. Sequence-Level Anomaly Detection ---
    logger.info(f"\nFitting One-Class SVM (nu={args.nu_param}) on Deep Kernels...")
    metrics = evaluate_novelty_detector(
        K_train=K_train, 
        K_test=K_test, 
        y_test_true=y_test_true_seq, 
        nu=args.nu_param
    )
    
    # --- 5. True Patient-Level Anomaly Aggregation ---
    patient_auc = evaluate_patient_level_novelty(metrics['anomaly_scores'], test_files_info, logger)
    
    elapsed = time.time() - start_time
    
    # --- 6. Output Results ---
    logger.info("\n=====================================================")
    logger.info(" FINAL RESULTS: PATIENT-LEVEL MIL EVALUATION (DEEP)")
    logger.info("=====================================================")
    logger.info(f"Execution Time          : {elapsed:.2f} seconds")
    logger.info(f"Sequence-Level ROC-AUC  : {metrics['auc']:.4f} (Expected to be low/noisy)")
    logger.info(f"PATIENT-LEVEL ROC-AUC   : {patient_auc:.4f} (True Clinical Metric)")
    logger.info("=====================================================")

if __name__ == "__main__":
    main()