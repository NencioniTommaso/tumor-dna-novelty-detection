"""
gram.py
Gram matrix computation for symmetric (training) and asymmetric (inference) kernels.
Handles Multiple Kernel Learning (MKL) weight generation, block-parallel matrix
multiplication, and kernel normalization.

Depends on features.py for sparse feature extraction.
"""

import logging
import os

import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from typing import List, Tuple, Optional

from src.mismatch import EPIGENETIC_ALPHABET
from src.features import extract_features_weighted

# Configure the module-level logger
logger = logging.getLogger(__name__)


def configure_single_threaded_blas():
    """
    Pin BLAS/MKL/OpenBLAS to single-threaded to avoid hidden thread contention
    with joblib. Call before parallel kernel computation.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"


def generate_mkl_weights(max_k: int, noise_threshold: int = 2, scaling: str = 'linear') -> list[float]:
    """
    Generates normalized Multiple Kernel Learning weights while suppressing short noisy k-mers.
    """
    weights = []

    for k in range(1, max_k + 1):
        if k <= noise_threshold:
            weights.append(0.0)
            continue

        base_val = k - noise_threshold
        if scaling == 'linear':
            weights.append(float(base_val))
        elif scaling == 'quadratic':
            weights.append(float(base_val ** 2))
        else:
            raise ValueError(f"Unsupported scaling strategy: {scaling}")

    total = sum(weights)
    if total == 0:
        return [0.0] * max_k

    return [round(weight / total, 4) for weight in weights]


def ensure_mkl_weights(max_k: int, mismatches: int, mkl_weights: Optional[List[float]]) -> List[float]:
    if mkl_weights is not None:
        return mkl_weights

    noise_threshold = max(1, 2 * mismatches)
    return generate_mkl_weights(max_k, noise_threshold=noise_threshold)


def normalize_gram(K: np.ndarray) -> np.ndarray:
    """
    Normalizes a dense Gram Matrix using efficient NumPy broadcasting.
    K_norm(i, j) = K(i, j) / sqrt(K(i, i) * K(j, j))
    """
    logger.debug("Normalizing dense Gram matrix...")
    diag = np.diag(K)
    diag_safe = np.maximum(diag, 1e-12)
    inv_sqrt_diag = 1.0 / np.sqrt(diag_safe)
    return K * inv_sqrt_diag[:, None] * inv_sqrt_diag[None, :]


# ====================================================================
# SYMMETRIC GRAM MATRIX (TRAINING)
# ====================================================================

def _compute_gram_block_pair(X_csr: sp.csr_matrix, r_start: int, r_end: int, c_start: int, c_end: int):
    """
    Computes a rectangular block intersection of the Gram matrix: K[r_start:r_end, c_start:c_end]
    """
    block_val = X_csr[r_start:r_end].dot(X_csr[c_start:c_end].T).toarray()
    return r_start, r_end, c_start, c_end, block_val


def _extract_and_compute_gram_k_symmetric(
    sequences: List[str], 
    k: int, 
    m: int, 
    weight: float,
    n_inner_jobs: int = -1,
    block_size: int = 1500
) -> Tuple[np.ndarray, dict]:
    
    if weight == 0.0:
        return np.zeros((len(sequences), len(sequences)), dtype=np.float64), {}

    logger.debug(f"Extracting features for k={k}, m={m} (weight={weight})...")
    X_k, vocab = extract_features_weighted(sequences, k, m, mismatch_decay=0.5, n_jobs=n_inner_jobs)
    X_k = X_k.multiply(np.sqrt(weight))
    
    diag_train = np.array(X_k.multiply(X_k).sum(axis=1)).flatten()
    train_state = {
        'k': k,
        'vocabulary': vocab,
        'X_train': X_k,
        'diag_train': diag_train
    }

    N = X_k.shape[0]
    K = np.zeros((N, N), dtype=np.float64)
    
    if N <= block_size:
        logger.debug(f"Computing exact Gram matrix directly for N={N} (k={k})")
        K_full = X_k.dot(X_k.T).toarray()
        return K_full, train_state
    
    ranges = [(i, min(i + block_size, N)) for i in range(0, N, block_size)]
    
    tasks = []
    for i, (r_start, r_end) in enumerate(ranges):
        for j, (c_start, c_end) in enumerate(ranges):
            if j >= i: 
                tasks.append((r_start, r_end, c_start, c_end))
                
    logger.debug(f"Executing {len(tasks)} block-pair multiplications for k={k} using {n_inner_jobs} threads...")
    results = Parallel(n_jobs=n_inner_jobs, prefer="threads")(
        delayed(_compute_gram_block_pair)(X_k, rs, re, cs, ce)
        for rs, re, cs, ce in tasks
    )
    
    for rs, re, cs, ce, block_val in results:
        K[rs:re, cs:ce] = block_val
        if rs != cs:
            K[cs:ce, rs:re] = block_val.T
            
    return K, train_state


def mixed_string_kernel(
    sequences: List[str], 
    k_max: int, 
    m: int = 0, 
    weights: Optional[List[float]] = None,
    n_jobs: int = -1
) -> Tuple[np.ndarray, dict]:
    
    configure_single_threaded_blas()

    if weights is None:
        weights = [1.0] * k_max

    start_k = max(1, m + 1) if m > 0 else 1
    active_ks = [k for k in range(start_k, k_max + 1) if weights[k-1] != 0.0]
    
    logger.info(f"Computing Mixed String Kernel for {len(sequences)} sequences...")
    logger.info(f"Active K-mers: {active_ks} | Sequential over k, all {n_jobs} cores to inner Gram computation")

    per_k_results = []
    for k in active_ks:
        result = _extract_and_compute_gram_k_symmetric(
            sequences, k, m, weights[k-1],
            n_inner_jobs=n_jobs
        )
        per_k_results.append(result)

    logger.info("Fusing sub-grams into final kernel...")
    sub_grams = [res[0] for res in per_k_results]
    train_states = {res[1]['k']: res[1] for res in per_k_results if res[1]}
    
    return sum(sub_grams), train_states


# ====================================================================
# ASYMMETRIC GRAM MATRIX (INFERENCE / CALIBRATION)
# ====================================================================

def _compute_asymmetric_block_pair(X_test: sp.csr_matrix, X_train: sp.csr_matrix, r_start: int, r_end: int, c_start: int, c_end: int):
    """
    Computes a rectangular block intersection of the Asymmetric Gram matrix: 
    X_test[r_start:r_end] * X_train[c_start:c_end]^T
    """
    block_val = X_test[r_start:r_end].dot(X_train[c_start:c_end].T).toarray()
    return r_start, r_end, c_start, c_end, block_val


def _extract_and_compute_asymmetric_k(
    test_seqs: List[str], 
    train_state_k: dict, 
    k: int, 
    m: int, 
    weight: float, 
    n_inner_jobs: int = -1, 
    block_size: int = 1500
):
    num_test = len(test_seqs)
    
    if weight == 0.0 or train_state_k is None:
        num_train = train_state_k['X_train'].shape[0] if train_state_k else 0
        return (np.zeros((num_test, num_train), dtype=np.float64), 
                np.zeros(num_test, dtype=np.float64), 
                np.zeros(num_train, dtype=np.float64) if num_train > 0 else None)

    X_train = train_state_k['X_train']
    vocab = train_state_k['vocabulary']
    diag_train_part = train_state_k['diag_train']
    num_train = X_train.shape[0]

    logger.debug(f"Extracting asymmetric features for k={k}, m={m} (weight={weight})...")
    
    # 1. Compute Cross-Terms using Train Vocabulary
    X_test_cross, _ = extract_features_weighted(test_seqs, k=k, m=m, mismatch_decay=0.5, vocabulary=vocab, n_jobs=n_inner_jobs)
    X_test_cross = X_test_cross.multiply(np.sqrt(weight))

    # Optimization: if the training vocabulary is the full permutation of the alphabet, 
    # then X_test_cross contains all possible features for the test sequences.
    # We can compute the diagonal directly from X_test_cross, saving a redundant extraction.
    is_full_vocab = len(vocab) == (len(EPIGENETIC_ALPHABET) ** k)

    if is_full_vocab:
        diag_test_part = np.array(X_test_cross.multiply(X_test_cross).sum(axis=1)).flatten()
    else:
        # 2. Compute Test Self-Norm exactly using its own vocabulary
        X_test_self, _ = extract_features_weighted(test_seqs, k, m, mismatch_decay=0.5, n_jobs=n_inner_jobs)
        X_test_self = X_test_self.multiply(np.sqrt(weight))
        diag_test_part = np.array(X_test_self.multiply(X_test_self).sum(axis=1)).flatten()

    K_part = np.zeros((num_test, num_train), dtype=np.float64)

    if num_test * num_train <= block_size * block_size:
        K_part = X_test_cross.dot(X_train.T).toarray()
    else:
        test_ranges = [(i, min(i + block_size, num_test)) for i in range(0, num_test, block_size)]
        train_ranges = [(i, min(i + block_size, num_train)) for i in range(0, num_train, block_size)]

        tasks = [(rs, re, cs, ce) for rs, re in test_ranges for cs, ce in train_ranges]

        results_blocks = Parallel(n_jobs=n_inner_jobs, prefer="threads")(
            delayed(_compute_asymmetric_block_pair)(X_test_cross, X_train, rs, re, cs, ce) 
            for rs, re, cs, ce in tasks
        )

        for rs, re, cs, ce, block_val in results_blocks:
            K_part[rs:re, cs:ce] = block_val

    return K_part, diag_test_part, diag_train_part


def compute_asymmetric_normalized_kernel(
    test_seqs: List[str],
    train_states: dict,
    max_k: int,
    mismatches: int,
    mkl_weights: List[float],
    n_jobs: int = -1,
) -> np.ndarray:
    
    configure_single_threaded_blas()

    num_test = len(test_seqs)
    # Determine num_train from any valid train_state
    num_train = 0
    for k_val, state in train_states.items():
        if state is not None:
            num_train = state['X_train'].shape[0]
            break
            
    if num_train == 0:
        raise ValueError("Invalid train_states dictionary provided.")
    
    K_cross = np.zeros((num_test, num_train), dtype=np.float64)
    diag_test = np.zeros(num_test, dtype=np.float64)
    diag_train = np.zeros(num_train, dtype=np.float64)

    active_ks = [k for k in range(1, max_k + 1) if mkl_weights[k - 1] != 0.0]

    if not active_ks:
        return K_cross 

    logger.info(f"Computing Asymmetric Kernel for {num_test}x{num_train} | Sequential over k, all {n_jobs} cores to inner computation")

    per_k_results = []
    for k in active_ks:
        result = _extract_and_compute_asymmetric_k(
            test_seqs, train_states.get(k), k, mismatches, mkl_weights[k-1], n_inner_jobs=n_jobs
        )
        per_k_results.append(result)

    logger.info("Fusing asymmetric sub-grams and normalizing...")
    for K_part, dt_part, dtr_part in per_k_results:
        K_cross += K_part
        diag_test += dt_part
        diag_train += dtr_part

    diag_test_safe = np.maximum(diag_test, 1e-12)
    diag_train_safe = np.maximum(diag_train, 1e-12)

    inv_sqrt_test = 1.0 / np.sqrt(diag_test_safe)
    inv_sqrt_train = 1.0 / np.sqrt(diag_train_safe)

    K_cross_norm = K_cross * inv_sqrt_test[:, None] * inv_sqrt_train[None, :]

    return K_cross_norm
