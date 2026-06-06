"""
features.py
Sparse feature extraction from DNA sequences using k-mer counting and
mismatch neighborhood expansion.

Produces scipy sparse matrices suitable for Gram matrix computation.
Depends on mismatch.py for neighborhood generation.
"""

import logging
import os

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from joblib import Parallel, delayed
from typing import List, Tuple, Optional

from src.mismatch import (
    EPIGENETIC_ALPHABET,
    generate_mismatch_neighborhood,
    generate_weighted_mismatch_neighborhood,
    build_full_vocabulary,
)

# Configure the module-level logger
logger = logging.getLogger(__name__)


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


def extract_features(sequences: List[str], k: int, m: int = 0, vocabulary: Optional[dict] = None) -> Tuple[sp.csr_matrix, dict]:
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
        return vectorizer.transform(sequences), vocabulary
        
    X = vectorizer.fit_transform(sequences)
    return X, vectorizer.vocabulary_


def _extract_chunk_weighted(
    chunk: List[str],
    chunk_offset: int,
    k: int,
    m: int,
    mismatch_decay: float,
    vocab: dict,
) -> Tuple[list, list, list]:
    """
    Processes a chunk of sequences with a fixed vocabulary.
    Each chunk is fully independent — no shared mutable state.
    Returns COO-format lists (rows, cols, vals) for sparse matrix assembly.
    """
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    for local_idx, sequence in enumerate(chunk):
        raw_kmers = [sequence[i : i + k] for i in range(len(sequence) - k + 1)]

        feature_weights: dict[int, float] = {}

        for raw_kmer in raw_kmers:
            neighbors = generate_weighted_mismatch_neighborhood(
                raw_kmer, m=m, alphabet=EPIGENETIC_ALPHABET
            )
            for neighbor, dist in neighbors:
                if neighbor in vocab:
                    col_idx = vocab[neighbor]
                    weight = mismatch_decay ** dist
                    feature_weights[col_idx] = feature_weights.get(col_idx, 0.0) + weight

        for col_idx, w in feature_weights.items():
            rows.append(chunk_offset + local_idx)
            cols.append(col_idx)
            vals.append(w)

    return rows, cols, vals


def extract_features_weighted(
    sequences: List[str],
    k: int,
    m: int = 0,
    mismatch_decay: float = 0.5,
    vocabulary: Optional[dict] = None,
    n_jobs: int = 1,
) -> Tuple[sp.csr_matrix, dict]:
    """
    Extracts weighted mismatch features. For each observed k-mer, its neighbors
    at Hamming distance d contribute weight = mismatch_decay^d.

    - mismatch_decay=1.0 is equivalent to extract_features (standard mismatch kernel).
    - mismatch_decay=0.0 counts only exact matches (equivalent to m=0).
    - Typical values: 0.3–0.7 to penalize inexact matches.

    Uses parallel chunking with a pre-enumerated vocabulary for multi-core execution.
    When vocabulary is None (training), the full alphabet vocabulary is pre-built.
    When vocabulary is provided (inference), it is used directly.

    Returns the same types as extract_features (csr_matrix, vocab dict), so it
    is a drop-in replacement compatible with the MKL and normalization pipeline.
    """
    # Fast path: if decay is 1.0 or no mismatches, delegate to standard extraction
    if m == 0 or mismatch_decay == 1.0:
        return extract_features(sequences, k, m, vocabulary)

    # Use provided vocabulary or pre-enumerate all possible k-mers
    if vocabulary is not None:
        vocab = vocabulary
    else:
        vocab = build_full_vocabulary(k)

    n_seqs = len(sequences)
    effective_jobs = min(n_jobs if n_jobs > 0 else (os.cpu_count() or 1), n_seqs)

    if effective_jobs <= 1:
        # Single-threaded fast path — avoid joblib overhead
        all_rows, all_cols, all_vals = _extract_chunk_weighted(
            sequences, 0, k, m, mismatch_decay, vocab
        )
    else:
        # Split sequences into chunks for parallel processing
        chunk_boundaries = np.array_split(range(n_seqs), effective_jobs)
        chunks = [(sequences[idx[0]:idx[-1]+1], idx[0]) for idx in chunk_boundaries if len(idx) > 0]

        logger.debug(f"Parallel feature extraction: k={k}, {len(chunks)} chunks across {effective_jobs} workers")

        results = Parallel(n_jobs=effective_jobs)(
            delayed(_extract_chunk_weighted)(chunk, offset, k, m, mismatch_decay, vocab)
            for chunk, offset in chunks
        )

        # Merge COO data from all chunks
        all_rows: list[int] = []
        all_cols: list[int] = []
        all_vals: list[float] = []
        for chunk_rows, chunk_cols, chunk_vals in results:
            all_rows.extend(chunk_rows)
            all_cols.extend(chunk_cols)
            all_vals.extend(chunk_vals)

    n_features = len(vocab)
    X = sp.csr_matrix(
        (np.array(all_vals, dtype=np.float64), (np.array(all_rows), np.array(all_cols))),
        shape=(n_seqs, n_features),
    )

    return X, vocab
