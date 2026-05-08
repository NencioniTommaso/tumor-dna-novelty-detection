"""
kernels.py
Contains sequence feature extraction, mismatch generation, and Gram matrix computation.
Optimized for multi-core execution using joblib with a Symmetric Block Strategy.

Upgraded with an Expanded Epigenetic Alphabet ('A', 'C', 'G', 'T', 'M', 'H') to capture
methylation states directly in the feature space as distinct structural variations.
"""

import itertools
import logging
import os
from functools import lru_cache

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from joblib import Parallel, delayed
from typing import List, Tuple, Optional

# Configure the module-level logger
logger = logging.getLogger(__name__)

# Expanded Epigenetic Alphabet
EPIGENETIC_ALPHABET = ('A', 'C', 'G', 'T', 'M', 'H')


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


@lru_cache(maxsize=100000)
def generate_mismatch_neighborhood(kmer: str, m: int = 1, alphabet: Tuple[str, ...] = EPIGENETIC_ALPHABET) -> List[str]:
    """
    Generates all k-mers within 'm' mismatches of the given kmer.
    Uses the 6-letter epigenetic alphabet to generate states.
    """
    if m == 0:
        return [kmer]
        
    neighborhood = set([kmer])
    kmer_list = list(kmer)
    indices = list(range(len(kmer)))
    
    # Loop from 1 mismatch up to 'm' mismatches
    for num_mismatches in range(1, m + 1):
        for positions in itertools.combinations(indices, num_mismatches):
            for replacement_chars in itertools.product(alphabet, repeat=num_mismatches):
                is_true_mismatch = True
                for pos, char in zip(positions, replacement_chars):
                    if kmer_list[pos] == char:
                        is_true_mismatch = False
                        break
                
                if is_true_mismatch:
                    mutated_kmer = kmer_list.copy()
                    for pos, char in zip(positions, replacement_chars):
                        mutated_kmer[pos] = char
                    
                    neighborhood.add("".join(mutated_kmer))
                    
    return list(neighborhood)


def mismatch_analyzer(sequence: str, k: int, m: int = 1) -> List[str]:
    """
    Custom analyzer for CountVectorizer.
    Extracts raw k-mers and generates the mismatch neighborhood using the epigenetic alphabet.
    """
    raw_kmers = [sequence[i:i+k] for i in range(len(sequence)-k+1)]
    
    expanded_kmers = []
    for raw_kmer in raw_kmers:
        # Directly map to mismatch neighborhood (no IUPAC resolution needed)
        expanded_kmers.extend(generate_mismatch_neighborhood(raw_kmer, m=m, alphabet=EPIGENETIC_ALPHABET))
            
    return expanded_kmers


def extract_features(sequences: List[str], k: int, m: int = 0, vocabulary: Optional[dict] = None) -> sp.csr_matrix:
    """
    Extracts sequence features. 
    Accepts an optional fixed vocabulary to allow for stateless, chunked parallelization.
    """
    vectorizer = CountVectorizer(
        analyzer=lambda x: mismatch_analyzer(x, k=k, m=m), 
        lowercase=False,
        vocabulary=vocabulary  # Inject fixed vocabulary
    )
    
    if vocabulary is not None:
        return vectorizer.transform(sequences)
        
    return vectorizer.fit_transform(sequences)


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
) -> np.ndarray:
    
    if weight == 0.0:
        return np.zeros((len(sequences), len(sequences)), dtype=np.float64)

    logger.debug(f"Extracting features for k={k}, m={m} (weight={weight})...")
    X_k = extract_features(sequences, k, m)
    X_k = X_k.multiply(np.sqrt(weight))
    
    N = X_k.shape[0]
    K = np.zeros((N, N), dtype=np.float64)
    
    if N <= block_size:
        logger.debug(f"Computing exact Gram matrix directly for N={N} (k={k})")
        K_full = X_k.dot(X_k.T).toarray()
        return K_full
    
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
            
    return K


def mixed_string_kernel(
    sequences: List[str], 
    k_max: int, 
    m: int = 0, 
    weights: Optional[List[float]] = None,
    n_jobs: int = -1
) -> Tuple[np.ndarray, Optional[sp.csr_matrix]]:
    
    if weights is None:
        weights = [1.0] * k_max

    start_k = max(1, m + 1) if m > 0 else 1
    active_ks = [k for k in range(start_k, k_max + 1) if weights[k-1] != 0.0]
    
    total_cores = os.cpu_count() if n_jobs == -1 else n_jobs
    
    large_ks = [k for k in active_ks if k > 3]
    n_outer = max(1, len(large_ks))
    inner_cores = max(1, total_cores // n_outer)

    logger.info(f"Computing Mixed String Kernel for {len(sequences)} sequences...")
    logger.info(f"Active K-mers: {active_ks} | Cores allocated: {total_cores} (Outer jobs: {n_jobs}, Inner threads ~{inner_cores})")

    def _jobs_for_k(k):
        return inner_cores if k > 3 else 1

    sub_grams = Parallel(n_jobs=n_jobs)(
        delayed(_extract_and_compute_gram_k_symmetric)(
            sequences, k, m, weights[k-1], 
            n_inner_jobs=_jobs_for_k(k)
        )
        for k in active_ks
    )

    logger.info("Fusing sub-grams into final kernel...")
    return sum(sub_grams), None


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
# PART FOR CALIBRATION AND INFERENCE
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
    train_seqs: List[str], 
    k: int, 
    m: int, 
    weight: float, 
    n_inner_jobs: int = -1, 
    block_size: int = 1500
):
    num_test = len(test_seqs)
    num_train = len(train_seqs)

    if weight == 0.0:
        return (np.zeros((num_test, num_train), dtype=np.float64), 
                np.zeros(num_test, dtype=np.float64), 
                np.zeros(num_train, dtype=np.float64))

    logger.debug(f"Extracting asymmetric features for k={k}, m={m} (weight={weight})...")
    X_combined = extract_features(test_seqs + train_seqs, k=k, m=m)
    X_combined = X_combined.multiply(np.sqrt(weight))

    X_test = X_combined[:num_test, :]
    X_train = X_combined[num_test:, :]

    K_part = np.zeros((num_test, num_train), dtype=np.float64)

    if num_test * num_train <= block_size * block_size:
        K_part = X_test.dot(X_train.T).toarray()
    else:
        test_ranges = [(i, min(i + block_size, num_test)) for i in range(0, num_test, block_size)]
        train_ranges = [(i, min(i + block_size, num_train)) for i in range(0, num_train, block_size)]

        tasks = [(rs, re, cs, ce) for rs, re in test_ranges for cs, ce in train_ranges]

        results_blocks = Parallel(n_jobs=n_inner_jobs, prefer="threads")(
            delayed(_compute_asymmetric_block_pair)(X_test, X_train, rs, re, cs, ce) 
            for rs, re, cs, ce in tasks
        )

        for rs, re, cs, ce, block_val in results_blocks:
            K_part[rs:re, cs:ce] = block_val

    diag_test_part = np.array(X_test.multiply(X_test).sum(axis=1)).flatten()
    diag_train_part = np.array(X_train.multiply(X_train).sum(axis=1)).flatten()

    return K_part, diag_test_part, diag_train_part


def compute_asymmetric_normalized_kernel(
    test_seqs: List[str],
    train_seqs: List[str],
    max_k: int,
    mismatches: int,
    mkl_weights: List[float],
    n_jobs: int = -1,
) -> np.ndarray:
    
    num_test = len(test_seqs)
    num_train = len(train_seqs)
    
    K_cross = np.zeros((num_test, num_train), dtype=np.float64)
    diag_test = np.zeros(num_test, dtype=np.float64)
    diag_train = np.zeros(num_train, dtype=np.float64)

    active_ks = [k for k in range(1, max_k + 1) if mkl_weights[k - 1] != 0.0]

    if not active_ks:
        return K_cross 

    total_cores = os.cpu_count() if n_jobs == -1 else n_jobs
    large_ks = [k for k in active_ks if k > 3]
    n_outer = max(1, len(large_ks))
    inner_cores = max(1, total_cores // n_outer)

    logger.info(f"Computing Asymmetric Kernel for {num_test}x{num_train} | Cores allocated: {total_cores}")

    def _jobs_for_k(k):
        return inner_cores if k > 3 else 1

    per_k_results = Parallel(n_jobs=n_jobs)(
        delayed(_extract_and_compute_asymmetric_k)(
            test_seqs, train_seqs, k, mismatches, mkl_weights[k-1], _jobs_for_k(k)
        )
        for k in active_ks
    )

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