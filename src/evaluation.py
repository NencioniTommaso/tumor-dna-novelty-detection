"""
evaluation.py
Handles the training, predicting, and evaluation of the anomaly detection model.
Optimized to support parallelized hyperparameter sweeps across CPU cores.
"""

import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, roc_auc_score
from joblib import Parallel, delayed
from typing import Dict, Any, List

def evaluate_novelty_detector(
    K_train: np.ndarray, 
    K_test: np.ndarray, 
    y_test_true: np.ndarray, 
    nu: float = 0.005
) -> Dict[str, Any]:
    """
    Fits a One-Class SVM on a precomputed training kernel and evaluates it on the test kernel.

    Args:
        K_train: Precomputed Gram matrix for training data (Train x Train).
        K_test: Precomputed Gram matrix for test data vs train data (Test x Train).
        y_test_true: Ground truth labels for the test set.
        nu: Upper bound on the fraction of training errors / lower bound on support vectors.

    Returns:
        A dictionary containing the nu parameter used, ROC-AUC score, and classification report.
    """
    # Initialize and fit the One-Class SVM
    oc_svm = OneClassSVM(kernel='precomputed', nu=nu)
    oc_svm.fit(K_train)
    
    # Generate predictions and raw decision function scores
    predictions = oc_svm.predict(K_test)
    anomaly_scores = oc_svm.decision_function(K_test)
    
    # Invert scores for ROC-AUC: 
    # OC-SVM decision function yields lower (negative) scores for anomalies.
    # roc_auc_score expects higher scores for the positive class (which we define as Cancer / -1).
    auc = roc_auc_score(y_test_true == -1, -anomaly_scores)
    
    # zero_division=0 prevents warnings if a model predicts purely one class
    report_str = classification_report(
        y_test_true, 
        predictions, 
        target_names=['Cancer (-1)', 'Healthy (1)'],
        zero_division=0
    )
    
    return {
        "nu": nu,
        "auc": auc,
        "report_str": report_str,
        "predictions": predictions,
        "anomaly_scores": anomaly_scores
    }

def parallel_evaluate_nu_grid(
    K_train: np.ndarray, 
    K_test: np.ndarray, 
    y_test_true: np.ndarray, 
    nu_grid: List[float],
    n_jobs: int = -1
) -> List[Dict[str, Any]]:
    """
    Evaluates multiple `nu` hyperparameters in parallel to find the best configuration.
    
    Since the underlying libsvm fit() operation is strictly single-threaded, 
    we achieve parallelism by dispatching independent model fits across CPU cores.

    Args:
        K_train: Precomputed Gram matrix for training data (Train x Train).
        K_test: Precomputed Gram matrix for test data vs train data (Test x Train).
        y_test_true: Ground truth labels for the test set.
        nu_grid: A list of `nu` values to evaluate (e.g., [0.001, 0.005, 0.01, 0.05]).
        n_jobs: Number of CPU cores to use (-1 uses all cores).

    Returns:
        A list of result dictionaries, sorted in descending order by AUC score.
    """
    # PARALLELIZATION: Map each nu value to a separate worker process
    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_novelty_detector)(K_train, K_test, y_test_true, nu)
        for nu in nu_grid
    )
    
    # Sort the results so the best performing hyperparameter configuration is at index 0
    results.sort(key=lambda x: x["auc"], reverse=True)
    
    return results