"""
calibrate_threshold.py
Runs a pretrained SVM on a validation cohort (Healthy + Tumor) to calculate 
the optimal clinical decision boundary using Youden's J statistic.
Updates the model artifact with this threshold.
"""

import os
import sys
import joblib
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.data_utils import load_tracked_patient_cohort
from src.kernels import compute_asymmetric_normalized_kernel, generate_mkl_weights
from src.model_io import load_svm_model
from experiments.experiments_utils import setup_logger, parse_arguments

logger = setup_logger(__name__)

def main():
    args = parse_arguments(project_root)
    model_path = os.path.join(project_root, "models", "ocsvm_pretrained.pkl")
    
    logger.info("=====================================================")
    logger.info(" PHASE 2: MODEL CALIBRATION & THRESHOLD TUNING")
    logger.info("=====================================================")

    # 1. Load the Pretrained Model
    svm, train_sequences, max_k, mismatches, mkl_weights, _ = load_svm_model(model_path, logger)
    
    if mkl_weights is None:
        mkl_weights = generate_mkl_weights(max_k, noise_threshold=max(1, 2 * mismatches))

    # 2. Define the Validation Cohort (Must include BOTH Healthy and Tumor)
    # Ensure these are different from the ones used in train_and_save_svm.py!
    val_normal_files = [os.path.join(args.data_dir, f"Healthy_{i}_merged_subset_1200000.fa") for i in range(6, 8)]
    val_tumor_files = [os.path.join(args.data_dir, f"Colo_{i}_merged_subset_1200000.fa") for i in range(1, 11) if i != 9]
    
    logger.info("Loading validation cohort...")
    # Notice we pass empty brackets for the training files, because we already have train_sequences!
    _, val_data, _, val_files_info = load_tracked_patient_cohort(
        [], val_normal_files, val_tumor_files, args, logger
    )
    
    # 3. Compute Inference Kernel for the Validation Set
    logger.info(f"Computing asymmetric kernel for {len(val_data)} validation sequences...")
    K_val = compute_asymmetric_normalized_kernel(
        test_seqs=val_data,
        train_seqs=train_sequences,
        max_k=max_k,
        mismatches=mismatches,
        mkl_weights=mkl_weights
    )
    
    # 4. Generate Sequence Scores
    logger.info("Generating anomaly scores...")
    anomaly_scores = svm.decision_function(K_val)
    
    # 5. Aggregate to Patient Level
    patient_y_true = []
    patient_scores = []
    current_idx = 0
    
    for info in val_files_info:
        num_seqs = info['num_sequences']
        seq_scores = anomaly_scores[current_idx : current_idx + num_seqs]
        current_idx += num_seqs
        
        inverted_scores = -seq_scores
        top_k = max(1, int(num_seqs * 0.05)) 
        patient_score = float(np.mean(np.sort(inverted_scores)[-top_k:]))
        
        patient_y_true.append(info['label'])
        patient_scores.append(patient_score)
        
        status = "TUMOR" if info['label'] == -1 else "HEALTHY"
        logger.info(f"[{status}] {info['filename']} -> Score: {patient_score:.4f}")

    # 6. Calculate Youden's J Statistic
    true_binary = (np.array(patient_y_true) == -1).astype(int) 
    fpr, tpr, thresholds = roc_curve(true_binary, patient_scores)
    youden_j = tpr - fpr 
    
    best_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[best_idx]
    val_auc = roc_auc_score(true_binary, patient_scores)
    
    logger.info("\n=====================================================")
    logger.info(f" VALIDATION ROC-AUC  : {val_auc:.4f}")
    logger.info(f" OPTIMAL THRESHOLD   : {optimal_threshold:.4f}")
    logger.info("=====================================================\n")
    
    # 7. Update and Save the Model Artifact
    logger.info("Updating model artifact with the calibrated threshold...")
    saved_state = joblib.load(model_path)
    saved_state['optimal_threshold'] = optimal_threshold
    joblib.dump(saved_state, model_path)
    
    logger.info("Calibration complete. Pipeline is ready for production inference.")

if __name__ == "__main__":
    main()