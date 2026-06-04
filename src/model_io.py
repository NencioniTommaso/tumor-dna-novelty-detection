"""
model_io.py
Persistence helpers for saving and loading precomputed-kernel OC-SVM artifacts.
"""

import logging
import os

import joblib

logger = logging.getLogger(__name__)


def save_svm_model(svm, train_sequences, max_k, mismatches, nu_param, save_path, mkl_weights=None, train_states=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    state = {
        'model': svm,
        'train_sequences': train_sequences,
        'max_k': max_k,
        'mismatches': mismatches,
        'nu_param': nu_param,
        'mkl_weights': mkl_weights,
        'train_states': train_states,
    }

    logger.info(f"Saving SVM state to {save_path}...")
    joblib.dump(state, save_path)
    logger.info("Model saved successfully!")


def load_svm_model(model_path):
    """
    Load a precomputed-kernel OC-SVM artifact.

    Returns (svm, train_sequences, max_k, mismatches, mkl_weights, train_states).
    """
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        raise FileNotFoundError(model_path)

    logger.info(f"Loading SVM state from {model_path}...")
    saved_state = joblib.load(model_path)

    svm = saved_state.get('model')
    train_sequences = saved_state.get('train_sequences', [])
    max_k = saved_state.get('max_k')
    mismatches = saved_state.get('mismatches')
    mkl_weights = saved_state.get('mkl_weights')
    train_states = saved_state.get('train_states')

    return svm, train_sequences, max_k, mismatches, mkl_weights, train_states