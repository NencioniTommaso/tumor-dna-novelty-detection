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

def _extract_and_compute_gram_k(sequences: List[str], k: int, m: int, weight: float) -> np.ndarray:
    """
    Helper function to extract features for a specific k-mer length, scale them, 
    and IMMEDIATELY compute the sub-Gram matrix. 
    Designed to be run in a separate process via joblib.
    """
    # OPTIMIZATION 1: Short-circuit if the biological weight is 0
    if weight == 0.0:
        return np.zeros((len(sequences), len(sequences)), dtype=np.float64)
        
    X_k = extract_features(sequences, k, m)
    X_k = X_k.multiply(np.sqrt(weight))
    
    # Return the dense N x N Gram matrix directly, discarding the massive sparse features
    return X_k.dot(X_k.T).toarray()

def mixed_string_kernel(
    sequences: List[str], 
    k_max: int, 
    m: int = 0, 
    weights: Optional[List[float]] = None,
    n_jobs: int = -1
) -> Tuple[np.ndarray, Optional[sp.csr_matrix]]:
    """
    Computes the fused Gram matrix for k-mers from k=1 up to k_max in parallel.
    Optimized for low-RAM execution by summing sub-Gram matrices instead of hstacking features.
    """
    if weights is None:
        weights = [1.0] * k_max
        
    start_k = max(1, m + 1) if m > 0 else 1
    
    # Compute each k-mer sub-Gram matrix concurrently using Process-based parallelism
    sub_grams = Parallel(n_jobs=n_jobs)(
        delayed(_extract_and_compute_gram_k)(sequences, k, m, weights[k-1]) 
        for k in range(start_k, k_max + 1)
    )

    # OPTIMIZATION 2: Sum the (N x N) dense matrices. 
    # This keeps memory strictly bound to O(N^2) regardless of how massive the feature space gets.
    composite_matrix = sum(sub_grams)
    
    # We return None for the sparse matrix to save RAM, as it is not used in the evaluation pipeline
    return composite_matrix, None

def normalize_gram(K: np.ndarray) -> np.ndarray:
    """
    Normalizes a dense Gram Matrix using efficient NumPy broadcasting.
    K_norm(i, j) = K(i, j) / sqrt(K(i, i) * K(j, j))
    """
    diag = np.diag(K)
    diag_safe = np.maximum(diag, 1e-12)
    inv_sqrt_diag = 1.0 / np.sqrt(diag_safe)
    return K * inv_sqrt_diag[:, None] * inv_sqrt_diag[None, :]