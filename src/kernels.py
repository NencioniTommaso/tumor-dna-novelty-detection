"""
kernels.py
Contains sequence feature extraction, mismatch generation, and Gram matrix computation.
Optimized for multi-core execution using joblib.
"""

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from joblib import Parallel, delayed
from typing import List, Tuple, Optional

def generate_mismatch_neighborhood(kmer: str, m: int = 1, alphabet: List[str] = ['A', 'C', 'G', 'T', 'M']) -> List[str]:
    """Generates all k-mers within 'm' mismatches of the given kmer."""
    if m == 0:
        return [kmer]
        
    neighborhood = set([kmer])
    for i in range(len(kmer)):
        for char in alphabet:
            if char != kmer[i]:
                mismatch_kmer = kmer[:i] + char + kmer[i+1:]
                neighborhood.add(mismatch_kmer)
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

def _extract_and_scale_k(sequences: List[str], k: int, m: int, weight: float) -> sp.csr_matrix:
    """
    Helper function to extract features for a specific k-mer length and scale them.
    Designed to be run in a separate process via joblib.
    """
    X_k = extract_features(sequences, k, m)
    return X_k.multiply(np.sqrt(weight))

def mixed_string_kernel(
    sequences: List[str], 
    k_max: int, 
    m: int = 0, 
    weights: Optional[List[float]] = None,
    n_jobs: int = -1
) -> Tuple[np.ndarray, sp.csr_matrix]:
    """
    Computes the fused Gram matrix for k-mers from k=1 up to k_max in parallel.
    
    Args:
        sequences: List of string sequences.
        k_max: Maximum k-mer length.
        m: Number of allowed mismatches (0 = Spectrum Kernel, >0 = Mismatch Kernel).
        weights: Weights to scale each k-th sub-kernel.
        n_jobs: Number of CPU cores to use for parallel extraction (-1 uses all cores).
        
    Returns:
        composite_matrix: The dense fused Gram matrix.
        X_composite_sparse: The concatenated sparse feature matrix.
    """
    if weights is None:
        weights = [1.0] * k_max
        
    # If using mismatches (m>0), k=1 and k=2 with 1 mismatch are pure noise.
    start_k = max(1, m + 1) if m > 0 else 1
    
    # PARALLELIZATION: Compute each k-mer sub-kernel concurrently using Process-based parallelism
    all_sparse_features = Parallel(n_jobs=n_jobs)(
        delayed(_extract_and_scale_k)(sequences, k, m, weights[k-1]) 
        for k in range(start_k, k_max + 1)
    )

    # Reassemble the results returned from the separate worker processes
    X_composite_sparse = sp.hstack(all_sparse_features, format='csr')
    composite_matrix = X_composite_sparse.dot(X_composite_sparse.T).toarray()
    
    return composite_matrix, X_composite_sparse

def normalize_gram(K: np.ndarray) -> np.ndarray:
    """
    Normalizes a dense Gram Matrix using efficient NumPy broadcasting.
    K_norm(i, j) = K(i, j) / sqrt(K(i, i) * K(j, j))
    """
    diag = np.diag(K)
    diag_safe = np.maximum(diag, 1e-12)
    inv_sqrt_diag = 1.0 / np.sqrt(diag_safe)
    return K * inv_sqrt_diag[:, None] * inv_sqrt_diag[None, :]