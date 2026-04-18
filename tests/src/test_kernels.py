"""
test_kernels.py
Comprehensive unit and integration tests for the sequence anomaly detection kernels.
Run this using: pytest tests/test_kernels.py -v
"""

import numpy as np
import scipy.sparse as sp
import pytest

# Adjust import path if necessary based on your project structure
from src.kernels import (
    generate_mismatch_neighborhood,
    mismatch_analyzer,
    extract_features,
    _extract_and_compute_gram_k,
    mixed_string_kernel,
    normalize_gram
)

# --- FIXTURES ---
@pytest.fixture
def sample_sequences():
    return ["ATGCATGC", "GCATGCAT", "AAAAAA", "TTTTTT"]

@pytest.fixture
def custom_alphabet():
    return ('A', 'C', 'G', 'T', 'M')

# --- 1. UNIT TESTS: NEIGHBORHOOD GENERATION ---

def test_neighborhood_m0(custom_alphabet):
    """Test that m=0 returns only the original k-mer."""
    kmer = "ATG"
    neighborhood = generate_mismatch_neighborhood(kmer, m=0, alphabet=custom_alphabet)
    assert len(neighborhood) == 1
    assert neighborhood[0] == "ATG"

def test_neighborhood_m1(custom_alphabet):
    """
    Test m=1 mutational expansion.
    For 'AAA' with a 5-letter alphabet, we expect:
    1 original + (3 positions * 4 alternative letters) = 13 k-mers.
    """
    kmer = "AAA"
    neighborhood = generate_mismatch_neighborhood(kmer, m=1, alphabet=custom_alphabet)
    assert len(neighborhood) == 13
    assert "AAA" in neighborhood
    assert "MAA" in neighborhood
    assert "AMA" in neighborhood
    assert "AAM" in neighborhood

def test_neighborhood_m2(custom_alphabet):
    """
    Test m=2 mutational expansion to ensure itertools logic holds.
    For 'AA' with a 5-letter alphabet:
    1 original + (2 pos * 4 alt) for m=1 + (1 pair * 4*4 alt) for m=2
    = 1 + 8 + 16 = 25 k-mers.
    """
    kmer = "AA"
    neighborhood = generate_mismatch_neighborhood(kmer, m=2, alphabet=custom_alphabet)
    assert len(neighborhood) == 25
    assert len(set(neighborhood)) == 25  # Ensure no duplicates were generated

def test_lru_cache_memoization(custom_alphabet):
    """
    Test that the LRU cache prevents recalculation of seen k-mers.
    Proves that the O(1) memory retrieval optimization is working.
    """
    # 1. Clear the cache to ensure a clean state for this specific test
    generate_mismatch_neighborhood.cache_clear()
    
    kmer1 = "ATGC"
    kmer2 = "CGTA"
    
    # 2. First call with kmer1
    # The function has never seen this kmer, so it MUST calculate it (Cache Miss)
    _ = generate_mismatch_neighborhood(kmer1, m=1, alphabet=custom_alphabet)
    
    info_after_first_call = generate_mismatch_neighborhood.cache_info()
    assert info_after_first_call.misses == 1
    assert info_after_first_call.hits == 0
    
    # 3. Second call with the EXACT SAME kmer1
    # The function should intercept this and return the saved list (Cache Hit)
    _ = generate_mismatch_neighborhood(kmer1, m=1, alphabet=custom_alphabet)
    
    info_after_second_call = generate_mismatch_neighborhood.cache_info()
    assert info_after_second_call.misses == 1  # Misses did NOT increase!
    assert info_after_second_call.hits == 1    # We successfully hit the cache!
    
    # 4. Third call with a DIFFERENT kmer2
    # The function hasn't seen this one yet, so it must calculate it (Cache Miss)
    _ = generate_mismatch_neighborhood(kmer2, m=1, alphabet=custom_alphabet)
    
    info_after_third_call = generate_mismatch_neighborhood.cache_info()
    assert info_after_third_call.misses == 2   # Misses increased
    assert info_after_third_call.hits == 1     # Hits stayed the same

# --- 2. UNIT TESTS: FEATURE EXTRACTION ---

def test_mismatch_analyzer():
    """Test if a sequence is correctly split and expanded."""
    sequence = "ATGC"
    k = 3
    m = 1
    # 3-mers of "ATGC" are "ATG" and "TGC"
    expanded = mismatch_analyzer(sequence, k, m)
    
    # Using default alphabet (5 chars), m=1 for 3-mers = 13 variations each.
    # Total should be 26.
    assert len(expanded) == 26
    assert "ATG" in expanded
    assert "TGC" in expanded

# --- 3. INTEGRATION TESTS: GRAM MATRIX COMPUTATION ---

def test_extract_and_compute_gram_short_circuit(sample_sequences):
    """Test that a weight of 0.0 successfully bypasses computation to save CPU."""
    gram = _extract_and_compute_gram_k(sample_sequences, k=3, m=2, weight=0.0)
    assert isinstance(gram, np.ndarray)
    assert gram.shape == (4, 4)
    assert np.all(gram == 0.0)  # Matrix must be entirely zeros

def test_mixed_string_kernel_shape(sample_sequences):
    """Test the full parallel multi-kernel fusion pipeline."""
    K, sparse_output = mixed_string_kernel(
        sequences=sample_sequences,
        k_max=3,
        m=1,
        weights=[0.1, 0.4, 0.5],
        n_jobs=1  # Run synchronously for predictable testing
    )
    
    assert isinstance(K, np.ndarray)
    assert K.shape == (4, 4)
    # The new optimized pipeline returns None for the sparse matrix
    assert sparse_output is None
    
    # Gram matrix must be symmetric
    assert np.allclose(K, K.T)
    # Diagonal values should be strictly positive
    assert np.all(np.diag(K) > 0)

# --- 4. UNIT TESTS: NORMALIZATION ---

def test_normalize_gram():
    """Test SVDD equivalent normalization."""
    # Dummy non-normalized covariance matrix
    K = np.array([
        [10.0, 2.0],
        [2.0,  5.0]
    ])
    
    K_norm = normalize_gram(K)
    
    # Diagonal must be exactly 1.0
    assert np.allclose(np.diag(K_norm), [1.0, 1.0])
    
    # Off-diagonals must be correctly scaled: K_ij / sqrt(K_ii * K_jj)
    # K_12 = 2.0 / sqrt(10 * 5) = 2.0 / sqrt(50) ≈ 0.2828
    expected_off_diag = 2.0 / np.sqrt(50.0)
    assert np.isclose(K_norm[0, 1], expected_off_diag)
    assert np.isclose(K_norm[1, 0], expected_off_diag)
    
    # Values must be bounded between -1 and 1
    assert np.all(K_norm >= -1.0) and np.all(K_norm <= 1.0)