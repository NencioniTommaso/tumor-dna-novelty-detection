"""
evaluation.py
Handles the training, predicting, and evaluation of the anomaly detection model.
Pure compute — no plotting or visualization side effects.
"""

import logging
import re
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, roc_auc_score
from typing import Dict, Any, List, Tuple

# Configure the module-level logger
logger = logging.getLogger(__name__)

def evaluate_novelty_detector(
    K_train: np.ndarray, 
    K_test: np.ndarray, 
    y_test_true: np.ndarray, 
    nu: float = 0.005,
) -> Dict[str, Any]:
    """
    Fits a One-Class SVM on a precomputed training kernel and evaluates it on the test kernel.
    Returns the fitted model and raw anomaly scores.
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
    
    logger.debug(f"Evaluation complete for nu={nu} | AUC: {auc:.4f}")
    
    return {
        "nu": nu,
        "auc": auc,
        "report_str": report_str,
        "predictions": predictions,
        "anomaly_scores": anomaly_scores,
    }


def evaluate_novelty_detector_nystrom(
    Phi_train: np.ndarray,
    Phi_test: np.ndarray,
    y_test_true: np.ndarray,
    nu: float = 0.005,
) -> Dict[str, Any]:
    """
    Fits a One-Class SVM on Nyström features and evaluates it.

    Uses OneClassSVM(kernel='linear') on the Nyström feature map Φ,
    which implicitly approximates the kernel SVM on the full Gram matrix.
    Returns the same result dict as evaluate_novelty_detector.
    """
    logger.debug(f"Initializing One-Class SVM (kernel='linear') with nu={nu}")
    oc_svm = OneClassSVM(kernel='linear', nu=nu)

    logger.debug(f"Fitting SVM on Φ_train shape {Phi_train.shape}...")
    oc_svm.fit(Phi_train)

    logger.debug("Generating predictions and computing anomaly scores...")
    predictions = oc_svm.predict(Phi_test)
    anomaly_scores = oc_svm.decision_function(Phi_test)

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

    logger.debug(f"Evaluation complete for nu={nu} | AUC: {auc:.4f}")

    return {
        "nu": nu,
        "auc": auc,
        "report_str": report_str,
        "predictions": predictions,
        "anomaly_scores": anomaly_scores,
    }


def _short_patient_name(filename: str) -> str:
    """Shorten filename (e.g. Colo_6_merged_subset_1200000.fa -> Colo_6)."""
    parts = filename.split("_")
    return f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else filename


def evaluate_patient_level_novelty(
    anomaly_scores: np.ndarray,
    test_files_info: List[Dict[str, Any]],
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Aggregates sequence-level anomaly scores to patient level using mean.
    Returns (patient_auc, per_patient_data) — no plotting side effects.

    Parameters
    ----------
    anomaly_scores : np.ndarray
        Raw decision_function output from OC-SVM (lower = more anomalous).
    test_files_info : list of dict
        Each dict has keys: 'num_sequences', 'label', 'filename'.

    Returns
    -------
    patient_auc : float
        Patient-level ROC-AUC.
    per_patient_data : list of dict
        Each dict has keys: 'short_name', 'label', 'inverted_scores', 'mean_score'.
        Suitable for passing to plotting.generate_score_distribution_plots().
    """
    patient_y_true = []
    patient_scores = []
    per_patient_data = []
    current_idx = 0

    logger.info("\n--- Patient-Level Anomaly Aggregation ---")

    for info in test_files_info:
        num_seqs = info['num_sequences']
        seq_scores = anomaly_scores[current_idx: current_idx + num_seqs]
        current_idx += num_seqs

        # Invert so higher = more anomalous
        inverted = -np.asarray(seq_scores)
        patient_score = float(np.mean(inverted))

        patient_y_true.append(info['label'])
        patient_scores.append(patient_score)

        short_name = _short_patient_name(info['filename'])
        per_patient_data.append({
            'short_name': short_name,
            'label': info['label'],
            'inverted_scores': inverted,
            'mean_score': patient_score,
        })

        status = "TUMOR" if info['label'] == -1 else "HEALTHY"
        logger.info(f"[{status}] {short_name} -> Mean Anomaly Score: {patient_score:.4f}")

    patient_auc = roc_auc_score(np.array(patient_y_true) == -1, patient_scores)

    return patient_auc, per_patient_data