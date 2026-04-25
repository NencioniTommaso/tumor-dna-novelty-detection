"""
kernels.py
Contains sequence feature extraction, mismatch generation, and Gram matrix computation.
Optimized for multi-core execution using joblib with a Symmetric Block Strategy.

Upgraded with Combinatorial Expansion for IUPAC ambiguity codes, ensuring accurate
biological representation without feature space explosion.
"""

import itertools
from functools import lru_cache

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from joblib import Parallel, delayed
from typing import List, Tuple, Optional

# Standard IUPAC ambiguity mapping
IUPAC_MAP = {
    'A': ['A'], 'C': ['C'], 'G': ['G'], 'T': ['T'],
    'M': ['A', 'C'],
    'R': ['A', 'G'],
    'W': ['A', 'T'],
    'S': ['C', 'G'],
    'Y': ['C', 'T'],
    'K': ['G', 'T'],
    'V': ['A', 'C', 'G'],
    'H': ['A', 'C', 'T'],
    'D': ['A', 'G', 'T'],
    'B': ['C', 'G', 'T'],
    'N': ['A', 'C', 'G', 'T']
}

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

def extract_features(sequences: List[str], k: int, m: int = 0) -> sp.csr_matrix:
    """
    Extracts sequence features. 
    Even if m=0 (Spectrum Kernel), we route through the custom analyzer 
    to ensure IUPAC ambiguity codes are combinatorially resolved.
    """
    vectorizer = CountVectorizer(
        analyzer=lambda x: mismatch_analyzer(x, k=k, m=m), 
        lowercase=False
    )
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
    X_k = extract_features(sequences, k, m)
    X_k = X_k.multiply(np.sqrt(weight))
    
    N = X_k.shape[0]
    K = np.zeros((N, N), dtype=np.float64)
    
    # Fast path for small datasets
    if N <= block_size:
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
    
    import os
    total_cores = os.cpu_count() if n_jobs == -1 else n_jobs
    
    # Heuristic: assign more inner cores to larger k-values (they cost more)
    large_ks = [k for k in active_ks if k > 3]
    n_outer = max(1, len(large_ks))     # how many large-k jobs run "concurrently"
    inner_cores = max(1, total_cores // n_outer)

    def _jobs_for_k(k):
        return inner_cores if k > 3 else 1

    sub_grams = Parallel(n_jobs=n_jobs)(
        delayed(_extract_and_compute_gram_k_symmetric)(
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