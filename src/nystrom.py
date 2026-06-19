"""
nystrom.py
Nyström kernel approximation for scalable OC-SVM training.

Produces an explicit low-rank feature map Φ ∈ ℝ^(N×m) such that
Φ·Φ^T ≈ K_norm, the cosine-normalized mismatch string kernel.
This allows training OneClassSVM(kernel='linear') on Φ instead of
materializing the dense N×N Gram matrix, reducing memory from O(N²)
to O(N·m) and enabling training on 100k+ sequences.

Mathematical summary
--------------------
1. Extract per-k sparse features X_k and apply MKL weights:
      X_combined = hstack([√w₁·X₁, √w₂·X₂, …])
2. Row-normalize (≡ cosine kernel normalization):
      X_norm_i = X_combined_i / ‖X_combined_i‖
3. Select m landmark indices uniformly at random.
4. Compute the landmark kernel W = X_norm[landmarks]·X_norm[landmarks]^T   (m×m)
5. Eigendecompose W = VΛV^T, compute W^{-1/2} = V·Λ^{-1/2}·V^T
6. Project all data: Φ = C · W^{-1/2},  where C = X_norm · X_norm[landmarks]^T  (N×m)

Then Φ·Φ^T ≈ K_norm and a linear SVM on Φ approximates the kernel SVM.
"""

import dataclasses
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh
from joblib import Parallel, delayed

from src.features import extract_features_weighted
from src.gram import configure_single_threaded_blas

# Configure the module-level logger
logger = logging.getLogger(__name__)


# ====================================================================
# STATE CONTAINER
# ====================================================================

@dataclasses.dataclass
class NystromState:
    """Persistent state for projecting new data into the Nyström feature space.

    Stored inside ModelArtifact for inference-time test projection.
    Contains only the landmark features and the projection matrix —
    NOT the full training feature matrix.
    """
    X_landmarks_norm: sp.csr_matrix  # m × D_total (sparse)
    W_inv_sqrt: np.ndarray           # m × m (dense)
    per_k_vocabs: Dict[int, dict]    # {k: vocabulary_dict}
    mkl_weights: List[float]
    max_k: int
    mismatches: int
    n_components: int
    landmark_indices: np.ndarray     # stored for reproducibility


# ====================================================================
# COMBINED FEATURE MATRIX CONSTRUCTION
# ====================================================================

def build_combined_feature_matrix(
    sequences: List[str],
    k_max: int,
    mismatches: int,
    mkl_weights: List[float],
    n_jobs: int = -1,
) -> Tuple[sp.csr_matrix, Dict[int, dict]]:
    """Extracts per-k sparse features, applies MKL weights, and hstacks.

    Parameters
    ----------
    sequences : list of str
        Training DNA sequences.
    k_max : int
        Maximum k-mer size.
    mismatches : int
        Allowed mismatch distance.
    mkl_weights : list of float
        One weight per k (length k_max). Zero-weighted k values are skipped.
    n_jobs : int
        Parallelism for feature extraction.

    Returns
    -------
    X_combined : sp.csr_matrix, shape (N, D_total)
        Horizontally stacked, MKL-weighted sparse feature matrix.
    per_k_vocabs : dict
        Mapping {k: vocabulary_dict} for each active k.
    """
    configure_single_threaded_blas()

    start_k = max(1, mismatches + 1) if mismatches > 0 else 1
    active_ks = [k for k in range(start_k, k_max + 1) if mkl_weights[k - 1] != 0.0]

    if not active_ks:
        raise ValueError("No active k-mer sizes after MKL weight filtering.")

    blocks: List[sp.csr_matrix] = []
    per_k_vocabs: Dict[int, dict] = {}

    for k in active_ks:
        weight = mkl_weights[k - 1]
        logger.info(f"Extracting features for k={k}, m={mismatches} (weight={weight:.4f})...")
        X_k, vocab = extract_features_weighted(
            sequences, k, mismatches, mismatch_decay=0.5, n_jobs=n_jobs
        )
        X_k = X_k.multiply(np.sqrt(weight))
        blocks.append(X_k)
        per_k_vocabs[k] = vocab

    X_combined = sp.hstack(blocks, format='csr')
    logger.info(
        f"Combined feature matrix: {X_combined.shape[0]} × {X_combined.shape[1]} "
        f"(nnz={X_combined.nnz}, density={X_combined.nnz / np.prod(X_combined.shape):.4e})"
    )

    return X_combined, per_k_vocabs


def build_combined_test_features(
    test_sequences: List[str],
    per_k_vocabs: Dict[int, dict],
    max_k: int,
    mismatches: int,
    mkl_weights: List[float],
    n_jobs: int = -1,
) -> sp.csr_matrix:
    """Builds the combined feature matrix for test sequences using training vocabularies.

    Parameters
    ----------
    test_sequences : list of str
        New sequences to project.
    per_k_vocabs : dict
        {k: vocabulary_dict} from training (stored in NystromState).
    max_k, mismatches, mkl_weights : kernel parameters (must match training).
    n_jobs : int
        Parallelism for feature extraction.

    Returns
    -------
    X_test_combined : sp.csr_matrix, shape (N_test, D_total)
    """
    configure_single_threaded_blas()

    # Iterate in sorted order to match the column layout from training
    active_ks = sorted(per_k_vocabs.keys())
    blocks: List[sp.csr_matrix] = []

    for k in active_ks:
        weight = mkl_weights[k - 1]
        vocab = per_k_vocabs[k]
        logger.info(f"Extracting test features for k={k} (weight={weight:.4f})...")
        X_k, _ = extract_features_weighted(
            test_sequences, k, mismatches, mismatch_decay=0.5,
            vocabulary=vocab, n_jobs=n_jobs
        )
        X_k = X_k.multiply(np.sqrt(weight))
        blocks.append(X_k)

    return sp.hstack(blocks, format='csr')


# ====================================================================
# ROW-WISE L2 NORMALIZATION
# ====================================================================

def normalize_rows(X: sp.csr_matrix) -> Tuple[sp.csr_matrix, np.ndarray]:
    """Row-wise L2 normalization of a sparse matrix.

    Equivalent to cosine kernel normalization:
    K_norm(i,j) = K(i,j) / √(K(i,i)·K(j,j))  where K = X·X^T.

    Returns
    -------
    X_norm : sp.csr_matrix
        Row-normalized copy.
    norms : np.ndarray, shape (N,)
        Original row norms (before normalization).
    """
    norms = sp.linalg.norm(X, axis=1)
    norms_safe = np.maximum(norms, 1e-12)
    inv_norms = 1.0 / norms_safe
    D_inv = sp.diags(inv_norms)
    X_norm = (D_inv @ X).tocsr()
    return X_norm, norms


# ====================================================================
# NYSTRÖM FIT / TRANSFORM
# ====================================================================

def nystrom_fit(
    X_norm: sp.csr_matrix,
    n_components: int,
    seed: int,
    per_k_vocabs: Dict[int, dict],
    mkl_weights: List[float],
    max_k: int,
    mismatches: int,
) -> NystromState:
    """Select landmarks and compute the Nyström projection matrix.

    Parameters
    ----------
    X_norm : sp.csr_matrix, shape (N, D)
        Row-normalized training features.
    n_components : int
        Number of landmark points (m).
    seed : int
        Random seed for landmark selection.
    per_k_vocabs, mkl_weights, max_k, mismatches :
        Kernel parameters to store for test-time projection.

    Returns
    -------
    NystromState
        Contains everything needed to project new data.
    """
    N = X_norm.shape[0]
    if n_components >= N:
        logger.warning(
            f"n_components ({n_components}) >= N ({N}), clamping to {N - 1}"
        )
        n_components = N - 1

    # 1. Select landmarks uniformly at random
    rng = np.random.RandomState(seed)
    landmark_indices = rng.choice(N, n_components, replace=False)
    landmark_indices.sort()

    X_landmarks = X_norm[landmark_indices]

    # 2. Compute W (m × m landmark kernel)
    logger.info(f"Computing landmark kernel W ({n_components}×{n_components})...")
    W = (X_landmarks @ X_landmarks.T).toarray()

    # 3. Eigendecompose W for stable W^{-1/2}
    eigenvalues, eigenvectors = eigh(W)

    # Truncate tiny eigenvalues for numerical stability
    max_eig = np.max(np.abs(eigenvalues))
    threshold = 1e-12 * max_eig if max_eig > 0 else 1e-12
    valid = eigenvalues > threshold
    n_valid = int(np.sum(valid))

    if n_valid < n_components:
        logger.warning(
            f"Truncated {n_components - n_valid} near-zero eigenvalues "
            f"(threshold={threshold:.2e})"
        )

    # W^{-1/2} = V · Λ^{-1/2} · V^T  (only valid eigenvalues)
    inv_sqrt_eigenvalues = np.zeros_like(eigenvalues)
    inv_sqrt_eigenvalues[valid] = 1.0 / np.sqrt(eigenvalues[valid])
    W_inv_sqrt = (eigenvectors * inv_sqrt_eigenvalues) @ eigenvectors.T

    logger.info(
        f"Nyström fit complete: {n_valid}/{n_components} eigenvalues retained, "
        f"condition number ≈ {eigenvalues[valid][-1] / eigenvalues[valid][0]:.2e}"
    )

    return NystromState(
        X_landmarks_norm=X_landmarks,
        W_inv_sqrt=W_inv_sqrt,
        per_k_vocabs=per_k_vocabs,
        mkl_weights=mkl_weights,
        max_k=max_k,
        mismatches=mismatches,
        n_components=n_components,
        landmark_indices=landmark_indices,
    )


def _compute_C_block(
    X_block: sp.csr_matrix,
    X_landmarks: sp.csr_matrix,
    r_start: int,
    r_end: int,
):
    """Compute a row-block of the C matrix: X_block · X_landmarks^T.

    Same pattern as gram.py's _compute_gram_block_pair.
    """
    block_val = X_block.dot(X_landmarks.T).toarray()
    return r_start, r_end, block_val


def nystrom_transform(
    X_norm: sp.csr_matrix,
    state: NystromState,
    n_jobs: int = -1,
    block_size: int = 1500,
) -> np.ndarray:
    """Project normalized features into the Nyström feature space.

    Uses block-parallel computation for the C matrix (same pattern as
    gram.py's block-pair multiplication) to maintain full multi-core
    utilization on large datasets.

    Parameters
    ----------
    X_norm : sp.csr_matrix, shape (N, D)
        Row-normalized features (train or test).
    state : NystromState
        Fitted Nyström state containing landmarks and W^{-1/2}.
    n_jobs : int
        Number of parallel threads for block C computation. -1 uses all cores.
    block_size : int
        Row block size for parallel C computation (default: 1500, matching gram.py).

    Returns
    -------
    Phi : np.ndarray, shape (N, n_components)
        Dense Nyström feature map.
    """
    N = X_norm.shape[0]
    m = state.n_components
    logger.info(f"Computing Nyström projection ({N} × {m})...")

    # --- Compute C = X_norm · X_landmarks^T  (N × m) ---
    if N <= block_size:
        # Small N: single call, avoid joblib overhead
        C = X_norm.dot(state.X_landmarks_norm.T).toarray()
    else:
        # Large N: block-parallel, matching gram.py's pattern
        C = np.zeros((N, m), dtype=np.float64)
        ranges = [(i, min(i + block_size, N)) for i in range(0, N, block_size)]

        logger.debug(
            f"Block-parallel C computation: {len(ranges)} blocks × {m} landmarks "
            f"using {n_jobs} threads..."
        )
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_compute_C_block)(
                X_norm[r_start:r_end], state.X_landmarks_norm, r_start, r_end
            )
            for r_start, r_end in ranges
        )

        for r_start, r_end, block_val in results:
            C[r_start:r_end] = block_val

    # --- Project: Φ = C · W^{-1/2}  (N × m) ---
    Phi = C @ state.W_inv_sqrt

    return Phi


def nystrom_fit_transform(
    X_norm: sp.csr_matrix,
    n_components: int,
    seed: int,
    per_k_vocabs: Dict[int, dict],
    mkl_weights: List[float],
    max_k: int,
    mismatches: int,
    n_jobs: int = -1,
) -> Tuple[np.ndarray, NystromState]:
    """Convenience: fit the Nyström approximation and transform in one call.

    Returns
    -------
    Phi : np.ndarray, shape (N, n_components)
    state : NystromState
    """
    state = nystrom_fit(
        X_norm, n_components, seed,
        per_k_vocabs, mkl_weights, max_k, mismatches,
    )
    Phi = nystrom_transform(X_norm, state, n_jobs=n_jobs)
    return Phi, state


# ====================================================================
# END-TO-END TEST PROJECTION
# ====================================================================

def build_and_project_test_features(
    test_sequences: List[str],
    state: NystromState,
    n_jobs: int = -1,
) -> np.ndarray:
    """End-to-end test projection: extract → combine → normalize → Nyström → Φ_test.

    Parameters
    ----------
    test_sequences : list of str
        New DNA sequences to score.
    state : NystromState
        Fitted state from training.
    n_jobs : int
        Parallelism for feature extraction.

    Returns
    -------
    Phi_test : np.ndarray, shape (N_test, n_components)
    """
    logger.info(f"Projecting {len(test_sequences)} test sequences into Nyström space...")

    X_test = build_combined_test_features(
        test_sequences, state.per_k_vocabs,
        state.max_k, state.mismatches, state.mkl_weights,
        n_jobs=n_jobs,
    )
    X_test_norm, _ = normalize_rows(X_test)
    Phi_test = nystrom_transform(X_test_norm, state, n_jobs=n_jobs)

    return Phi_test
