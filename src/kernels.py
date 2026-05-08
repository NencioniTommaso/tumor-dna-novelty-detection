"""
kernels.py
Contains sequence feature extraction, mismatch generation, and Gram matrix computation.
Optimized for multi-core execution using joblib with a Symmetric Block Strategy.

Upgraded with Combinatorial Expansion for IUPAC ambiguity codes, ensuring accurate
biological representation without feature space explosion.
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

# Standard IUPAC ambiguity mapping
IUPAC_MAP = {
    'A': ['A'], 'C': ['C'], 'G': ['G'], 'T': ['T'],
    'M': ['A', 'C'], 'R': ['A', 'G'], 'W': ['A', 'T'],
    'S': ['C', 'G'], 'Y': ['C', 'T'], 'K': ['G', 'T'],
    'V': ['A', 'C', 'G'], 'H': ['A', 'C', 'T'], 
    'D': ['A', 'G', 'T'], 'B': ['C', 'G', 'T'],
    'N': ['A', 'C', 'G', 'T']
}

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
def resolve_ambiguous_kmer(kmer: str) -> List[str]:
    """
    Expands an ambiguous k-mer into all its exact biological possibilities.
    Example: 'ATM' -> ['ATA', 'ATC']
    """
    possible_bases = [IUPAC_MAP.get(char.upper(), ['N']) for char in kmer]
    return ["".join(combo) for combo in itertools.product(*possible_bases)]

@lru_cache(maxsize=100000)
def generate_mismatch_neighborhood(kmer: str, m: int = 1, alphabet: Tuple[str, ...] = ('A', 'C', 'G', 'T')) -> List[str]:
    """
    Generates all k-mers within 'm' mismatches of the given kmer.
    Upgraded to use itertools for exact mutational combinations and LRU Cache for speed.
    The alphabet strictly defaults to the 4 standard bases.
    """
    if m == 0:
        return [kmer]
        
    neighborhood = set([kmer])
    kmer_list = list(kmer)
    indices = list(range(len(kmer)))
    
    # We loop from 1 mismatch up to 'm' mismatches
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
    Custom analyzer for CountVectorizer to expand sequences:
    1. Extracts raw k-mers.
    2. Resolves biological ambiguities (e.g., M, R, Y) into standard bases.
    3. Generates the mismatch neighborhood using standard bases.
    """
    raw_kmers = [sequence[i:i+k] for i in range(len(sequence)-k+1)]
    
    expanded_kmers = []
    for raw_kmer in raw_kmers:
        # Step 1: Resolve IUPAC ambiguities first
        resolved_exact_kmers = resolve_ambiguous_kmer(raw_kmer)
        
        # Step 2: Apply mismatch generation ONLY to standard A,C,G,T strings
        for exact_kmer in resolved_exact_kmers:
            expanded_kmers.extend(generate_mismatch_neighborhood(exact_kmer, m=m, alphabet=('A', 'C', 'G', 'T')))
            
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
    
    # If vocab is fixed, we only need to transform (skips fitting overhead)
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

    # 1. Extract and scale features
    logger.debug(f"Extracting features for k={k}, m={m} (weight={weight})...")
    X_k = extract_features(sequences, k, m)
    X_k = X_k.multiply(np.sqrt(weight))
    
    N = X_k.shape[0]
    K = np.zeros((N, N), dtype=np.float64)
    
    # Fast path for small datasets
    if N <= block_size:
        logger.debug(f"Computing exact Gram matrix directly for N={N} (k={k})")
        K_full = X_k.dot(X_k.T).toarray()
        return K_full
    
    # 2. Define block ranges
    ranges = [(i, min(i + block_size, N)) for i in range(0, N, block_size)]
    
    # 3. Create tasks ONLY for the upper triangle of blocks (I <= J)
    tasks = []
    for i, (r_start, r_end) in enumerate(ranges):
        for j, (c_start, c_end) in enumerate(ranges):
            if j >= i: # Upper triangle of blocks
                tasks.append((r_start, r_end, c_start, c_end))
                
    # 4. Execute block-pair multiplications in parallel
    logger.debug(f"Executing {len(tasks)} block-pair multiplications for k={k} using {n_inner_jobs} threads...")
    results = Parallel(n_jobs=n_inner_jobs, prefer="threads")(
        delayed(_compute_gram_block_pair)(X_k, rs, re, cs, ce)
        for rs, re, cs, ce in tasks
    )
    
    # 5. Reassemble the symmetric matrix
    for rs, re, cs, ce, block_val in results:
        K[rs:re, cs:ce] = block_val
        # Mirror to the lower triangle if it's an off-diagonal block
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
    
    # Heuristic: assign more inner cores to larger k-values (they cost more)
    large_ks = [k for k in active_ks if k > 3]
    n_outer = max(1, len(large_ks))     # how many large-k jobs run "concurrently"
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


# PART FOR CALIBRATION AND INFERENCE

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
    """
    Module-level worker function to compute the cross-block and diagonals for a single k-mer.
    Because it is at the module level, joblib will not suffer from closure serialization overhead.
    """
    num_test = len(test_seqs)
    num_train = len(train_seqs)

    if weight == 0.0:
        return (np.zeros((num_test, num_train), dtype=np.float64), 
                np.zeros(num_test, dtype=np.float64), 
                np.zeros(num_train, dtype=np.float64))

    # 1. Extract combined features to ensure vocabulary alignment
    logger.debug(f"Extracting asymmetric features for k={k}, m={m} (weight={weight})...")
    X_combined = extract_features(test_seqs + train_seqs, k=k, m=m)
    X_combined = X_combined.multiply(np.sqrt(weight))

    # 2. Slice the sparse matrices cleanly
    X_test = X_combined[:num_test, :]
    X_train = X_combined[num_test:, :]

    K_part = np.zeros((num_test, num_train), dtype=np.float64)

    # Fast path for small data
    if num_test * num_train <= block_size * block_size:
        K_part = X_test.dot(X_train.T).toarray()
    else:
        # Define independent block ranges
        test_ranges = [(i, min(i + block_size, num_test)) for i in range(0, num_test, block_size)]
        train_ranges = [(i, min(i + block_size, num_train)) for i in range(0, num_train, block_size)]

        tasks = [(rs, re, cs, ce) for rs, re in test_ranges for cs, ce in train_ranges]

        # Execute block-pairs using inner threads
        results_blocks = Parallel(n_jobs=n_inner_jobs, prefer="threads")(
            delayed(_compute_asymmetric_block_pair)(X_test, X_train, rs, re, cs, ce) 
            for rs, re, cs, ce in tasks
        )

        # Assemble the partial asymmetric matrix
        for rs, re, cs, ce, block_val in results_blocks:
            K_part[rs:re, cs:ce] = block_val

    # 3. Compute diagonals rapidly using sparse row sums
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
    """
    Computes ONLY the Test vs Train block of the Gram matrix and normalizes it.
    Uses Hierarchical Parallelization identical to the training kernel.
    """
    num_test = len(test_seqs)
    num_train = len(train_seqs)
    
    K_cross = np.zeros((num_test, num_train), dtype=np.float64)
    diag_test = np.zeros(num_test, dtype=np.float64)
    diag_train = np.zeros(num_train, dtype=np.float64)

    active_ks = [k for k in range(1, max_k + 1) if mkl_weights[k - 1] != 0.0]

    if not active_ks:
        return K_cross # Failsafe

    total_cores = os.cpu_count() if n_jobs == -1 else n_jobs
    large_ks = [k for k in active_ks if k > 3]
    n_outer = max(1, len(large_ks))
    inner_cores = max(1, total_cores // n_outer)

    logger.info(f"Computing Asymmetric Kernel for {num_test}x{num_train} | Cores allocated: {total_cores}")

    def _jobs_for_k(k):
        return inner_cores if k > 3 else 1

    # Outer parallelism maps across active k-mers
    per_k_results = Parallel(n_jobs=n_jobs)(
        delayed(_extract_and_compute_asymmetric_k)(
            test_seqs, train_seqs, k, mismatches, mkl_weights[k-1], _jobs_for_k(k)
        )
        for k in active_ks
    )

    # Accumulate results
    logger.info("Fusing asymmetric sub-grams and normalizing...")
    for K_part, dt_part, dtr_part in per_k_results:
        K_cross += K_part
        diag_test += dt_part
        diag_train += dtr_part

    # Apply SVDD Normalization Formula
    diag_test_safe = np.maximum(diag_test, 1e-12)
    diag_train_safe = np.maximum(diag_train, 1e-12)

    inv_sqrt_test = 1.0 / np.sqrt(diag_test_safe)
    inv_sqrt_train = 1.0 / np.sqrt(diag_train_safe)

    K_cross_norm = K_cross * inv_sqrt_test[:, None] * inv_sqrt_train[None, :]

    return K_cross_norm