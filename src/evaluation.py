"""
evaluation.py
Handles the training, predicting, and evaluation of the anomaly detection model.
Optimized to support parallelized hyperparameter sweeps across CPU cores.
"""

import logging
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, roc_auc_score
from typing import Dict, Any

# Configure the module-level logger
logger = logging.getLogger(__name__)

def evaluate_novelty_detector(
    K_train: np.ndarray, 
    K_test: np.ndarray, 
    y_test_true: np.ndarray, 
    nu: float = 0.005,
    seq_fpr: float = 0.01
) -> Dict[str, Any]:
    """
    Fits a One-Class SVM on a precomputed training kernel and evaluates it on the test kernel.
    """
    logger.debug(f"Initializing One-Class SVM with nu={nu}")
    oc_svm = OneClassSVM(kernel='precomputed', nu=nu)
    
    logger.debug(f"Fitting SVM on K_train shape {K_train.shape}...")
    oc_svm.fit(K_train)
    
    logger.debug("Generating predictions and computing anomaly scores...")
    predictions = oc_svm.predict(K_test)
    anomaly_scores = oc_svm.decision_function(K_test)
    
    # Invert scores for ROC-AUC: 
    # OC-SVM decision function yields lower (negative) scores for anomalies.
    auc = roc_auc_score(y_test_true == -1, -anomaly_scores)
    
    # zero_division=0 prevents warnings if a model predicts purely one class
    report_str = classification_report(
        y_test_true, 
        predictions, 
        target_names=['Cancer (-1)', 'Healthy (1)'],
        zero_division=0
    )
    
    # Compute tau_seq on training set to enforce the desired FPR
    # The anomaly_scores are negative for anomalies, so we invert them so higher = more anomalous
    train_scores = -oc_svm.decision_function(K_train)
    tau_percentile = 100.0 * (1.0 - seq_fpr)
    tau_seq = float(np.percentile(train_scores, tau_percentile))
    
    logger.debug(f"Evaluation complete for nu={nu} | AUC: {auc:.4f} | tau_seq: {tau_seq:.4f}")
    
    return {
        "nu": nu,
        "auc": auc,
        "report_str": report_str,
        "predictions": predictions,
        "anomaly_scores": anomaly_scores,
        "tau_seq": tau_seq
    }


def compute_patient_score(seq_scores, tau_seq: float) -> float:
    """
    Computes the proportion of sequences that exceed the sequence-level threshold tau_seq.
    """
    inverted_scores = -np.asarray(seq_scores)
    return float(np.mean(inverted_scores > tau_seq))


def evaluate_patient_level_novelty(anomaly_scores, test_files_info, tau_seq: float, logger):
    patient_y_true = []
    patient_scores = []
    current_idx = 0

    logger.info("\n--- Patient-Level Anomaly Aggregation ---")

    for info in test_files_info:
        num_seqs = info['num_sequences']
        seq_scores = anomaly_scores[current_idx: current_idx + num_seqs]
        current_idx += num_seqs

        patient_score = compute_patient_score(seq_scores, tau_seq)

        patient_y_true.append(info['label'])
        patient_scores.append(patient_score)

        status = "TUMOR" if info['label'] == -1 else "HEALTHY"
        logger.info(f"[{status}] {info['filename']} -> Outlier Proportion: {patient_score:.4%}")

    patient_auc = roc_auc_score(np.array(patient_y_true) == -1, patient_scores)
    return patient_auc
    