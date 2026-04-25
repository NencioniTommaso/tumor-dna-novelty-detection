"""
evaluation.py
Handles the training, predicting, and evaluation of the anomaly detection model.
Optimized to support parallelized hyperparameter sweeps across CPU cores.
"""

import logging
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, roc_auc_score
from joblib import Parallel, delayed
from typing import Dict, Any, List

# Configure the module-level logger
logger = logging.getLogger(__name__)

def evaluate_novelty_detector(
    K_train: np.ndarray, 
    K_test: np.ndarray, 
    y_test_true: np.ndarray, 
    nu: float = 0.005
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
    
    logger.debug(f"Evaluation complete for nu={nu} | AUC: {auc:.4f}")
    
    return {
        "nu": nu,
        "auc": auc,
        "report_str": report_str,
        "predictions": predictions,
        "anomaly_scores": anomaly_scores
    }
    