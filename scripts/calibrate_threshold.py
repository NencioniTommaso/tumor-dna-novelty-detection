"""
calibrate_threshold.py
Runs a pretrained SVM on a validation cohort (Healthy + Tumor) to calculate 
the optimal clinical decision boundary using Youden's J statistic.
Updates the model artifact with this threshold.
"""

import os
import sys
import time
import joblib
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, classification_report

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.data_utils import load_tracked_patient_cohort
from src.kernels import compute_asymmetric_normalized_kernel, ensure_mkl_weights
from src.evaluation import compute_patient_score
from src.model_io import load_svm_model
from experiments.experiments_utils import (
    setup_logger,
    create_base_parser,
    add_data_dir_arg,
    add_cache_dir_arg,
    add_test_sampling_args,
    add_seed_arg,
    add_model_path_arg,
    build_validation_files,
)

logger = setup_logger(__name__)

def main():
    parser = create_base_parser("Calibrate a pretrained model and update decision threshold.")
    add_data_dir_arg(parser, required=True)
    add_cache_dir_arg(parser, project_root)
    add_test_sampling_args(parser)
    add_seed_arg(parser)
    add_model_path_arg(parser, project_root)
    args = parser.parse_args()
    model_path = args.model_path
    
    logger.info("=====================================================")
    logger.info(" PHASE 2: MODEL CALIBRATION & THRESHOLD TUNING")
    logger.info("=====================================================")

    start_time = time.perf_counter()

    # 1. Load the Pretrained Model
    svm, train_sequences, max_k, mismatches, mkl_weights, _, train_states, tau_seq = load_svm_model(model_path)
    
    mkl_weights = ensure_mkl_weights(max_k, mismatches, mkl_weights)

    # 2. Define the Validation Cohort (Must include BOTH Healthy and Tumor)
    # Ensure these are different from the ones used in train_and_save_svm.py!
    val_normal_files, val_tumor_files = build_validation_files(args.data_dir)
    
    logger.info("Loading validation cohort...")
    # Notice we pass empty brackets for the training files, because we already have train_sequences!
    _, val_data, _, val_files_info = load_tracked_patient_cohort(
        [], val_normal_files, val_tumor_files, 0, args.max_test_normal, args.max_test_tumor, args.seed, args.cache_dir, logger
    )
    
    # 3. Compute Inference Kernel for the Validation Set
    logger.info(f"Computing asymmetric kernel for {len(val_data)} validation sequences...")
    K_val = compute_asymmetric_normalized_kernel(
        test_seqs=val_data,
        train_states=train_states,
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
        
        patient_score = compute_patient_score(seq_scores, tau_seq)
        
        patient_y_true.append(info['label'])
        patient_scores.append(patient_score)
        
        status = "TUMOR" if info['label'] == -1 else "HEALTHY"
        logger.info(f"[{status}] {info['filename']} -> Score: {patient_score:.4f}")

    # 6. Calculate Youden's J Statistic using Stratified Bootstrap
    true_binary = (np.array(patient_y_true) == -1).astype(int) 
    patient_scores_arr = np.array(patient_scores)
    
    # Separate indices by class for stratified resampling
    idx_healthy = np.where(true_binary == 0)[0]
    idx_tumor = np.where(true_binary == 1)[0]
    
    if len(idx_healthy) == 0 or len(idx_tumor) == 0:
        raise ValueError(
            f"Calibration requires both healthy and tumor patients in the validation set. "
            f"Got {len(idx_healthy)} healthy, {len(idx_tumor)} tumor."
        )
    
    rng = np.random.default_rng(args.seed)
    collected_thresholds = []
    N = 100
    for _ in range(N):
        # Stratified bootstrap: resample within each class independently
        boot_healthy = rng.choice(idx_healthy, size=len(idx_healthy), replace=True)
        boot_tumor = rng.choice(idx_tumor, size=len(idx_tumor), replace=True)
        boot_idx = np.concatenate([boot_healthy, boot_tumor])
        
        resampled_binary = true_binary[boot_idx]
        resampled_scores = patient_scores_arr[boot_idx]
        
        fpr, tpr, thresholds = roc_curve(resampled_binary, resampled_scores)
        youden_j = tpr - fpr 
        best_idx = np.argmax(youden_j)
        collected_thresholds.append(thresholds[best_idx])
    
    collected_thresholds = np.array(collected_thresholds)
    optimal_threshold = np.median(collected_thresholds)
    
    ci_lower = np.percentile(collected_thresholds, 2.5)
    ci_upper = np.percentile(collected_thresholds, 97.5)
    
    val_auc = roc_auc_score(true_binary, patient_scores_arr)
    
    preds = (patient_scores_arr >= optimal_threshold).astype(int)
    report = classification_report(true_binary, preds, target_names=["Healthy", "Tumor"])
    logger.info("\n--- Validation Classification Report ---")
    logger.info("\n" + report)
    
    logger.info("\n=====================================================")
    logger.info(f" VALIDATION ROC-AUC  : {val_auc:.4f}")
    logger.info(f" OPTIMAL THRESHOLD   : {optimal_threshold:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
    logger.info("=====================================================\n")
    
    # 7. Update and Save the Model Artifact
    logger.info("Updating model artifact with the calibrated threshold...")
    saved_state = joblib.load(model_path)
    saved_state['optimal_threshold'] = optimal_threshold
    joblib.dump(saved_state, model_path)
    
    logger.info("Calibration complete. Pipeline is ready for production inference.")

    elapsed = time.perf_counter() - start_time
    logger.info(f"Calibration time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()