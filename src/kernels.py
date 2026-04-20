"""
kernels.py
Contains sequence feature extraction, mismatch generation, and Gram matrix computation.
Optimized for multi-core execution using joblib.
"""


import itertools
from functools import lru_cache

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from joblib import Parallel, delayed
from typing import List, Tuple, Optional

@lru_cache(maxsize=100000)
def generate_mismatch_neighborhood(kmer: str, m: int = 1, alphabet: Tuple[str, ...] = ('A', 'C', 'G', 'T', 'M')) -> List[str]:
    """
    Generates all k-mers within 'm' mismatches of the given kmer.
    Upgraded to use itertools for exact mutational combinations and LRU Cache for speed.
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
    """Custom analyzer for CountVectorizer to expand sequences into mutational neighborhoods."""
    kmers = [sequence[i:i+k] for i in range(len(sequence)-k+1)]
    expanded_kmers = []
    for kmer in kmers:
        expanded_kmers.extend(generate_mismatch_neighborhood(kmer, m=m))
    return expanded_kmers

def extract_features(sequences: List[str], k: int, m: int = 0) -> sp.csr_matrix:
    """
    Extracts sequence features using either exact k-mers (Spectrum, m=0) 
    or relaxed k-mers (Mismatch, m>0).
    """
    if m == 0:
        vectorizer = CountVectorizer(analyzer='char', ngram_range=(k, k), lowercase=False)
    else:
        vectorizer = CountVectorizer(
            analyzer=lambda x: mismatch_analyzer(x, k=k, m=m), 
            lowercase=False
        )
    return vectorizer.fit_transform(sequences)

def _compute_gram_block(X_csr: sp.csr_matrix, row_start: int, row_end: int) -> np.ndarray:
    """
    Computes a horizontal slice of the Gram matrix: K[row_start:row_end, :].
    X_csr must already be weighted. Each block is fully independent.
    """
    return X_csr[row_start:row_end].dot(X_csr.T).toarray()


def _extract_and_compute_gram_k(
    sequences: List[str], 
    k: int, 
    m: int, 
    weight: float,
    n_inner_jobs: int = 1,      # how many cores to use inside this k-task
    block_size: int = 2000       # rows per block for the inner parallelism
) -> np.ndarray:
    """
    Extracts features for a specific k (single pass), then computes the
    full sub-Gram matrix by parallelizing the dot product across row blocks.
    """
    if weight == 0.0:
        return np.zeros((len(sequences), len(sequences)), dtype=np.float64)

    # Feature extraction: done ONCE, O(N) — this does not change
    X_k = extract_features(sequences, k, m)
    X_k = X_k.multiply(np.sqrt(weight))
    
    N = X_k.shape[0]
    
    # If only one inner job (small k, fast anyway), skip the overhead
    if n_inner_jobs == 1 or N <= block_size:
        return X_k.dot(X_k.T).toarray()
    
    # Build row-block ranges
    ranges = [(i, min(i + block_size, N)) for i in range(0, N, block_size)]
    
    # Parallelize the dot product: each worker gets one row slice
    blocks = Parallel(n_jobs=n_inner_jobs, prefer="threads")(
        delayed(_compute_gram_block)(X_k, start, end)
        for start, end in ranges
    )
    
    return np.vstack(blocks)

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
    
    import os
    total_cores = os.cpu_count() if n_jobs == -1 else n_jobs
    
    # Heuristic: assign more inner cores to larger k-values (they cost more)
    # Small k (<=3): run on 1 core, they're fast enough
    # Large k (>3):  split remaining cores between them
    large_ks = [k for k in active_ks if k > 3]
    n_outer = max(1, len(large_ks))     # how many large-k jobs run "concurrently"
    inner_cores = max(1, total_cores // n_outer)

    def _jobs_for_k(k):
        return inner_cores if k > 3 else 1

    sub_grams = Parallel(n_jobs=n_jobs)(
        delayed(_extract_and_compute_gram_k)(
            sequences, k, m, weights[k-1], 
            n_inner_jobs=_jobs_for_k(k)
        )
        for k in active_ks
    )

    return sum(sub_grams), None

def normalize_gram(K: np.ndarray) -> np.ndarray:
    """
    Normalizes a dense Gram Matrix using efficient NumPy broadcasting.
    K_norm(i, j) = K(i, j) / sqrt(K(i, i) * K(j, j))
    """
    diag = np.diag(K)
    diag_safe = np.maximum(diag, 1e-12)
    inv_sqrt_diag = 1.0 / np.sqrt(diag_safe)
    return K * inv_sqrt_diag[:, None] * inv_sqrt_diag[None, :]