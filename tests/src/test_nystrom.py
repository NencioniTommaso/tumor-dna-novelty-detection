"""
test_nystrom.py
Comprehensive unit and integration tests for the Nyström kernel approximation.
Run this using: pytest tests/src/test_nystrom.py -v
"""

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.svm import OneClassSVM

from src.nystrom import (
    NystromState,
    build_combined_feature_matrix,
    build_combined_test_features,
    normalize_rows,
    nystrom_fit,
    nystrom_transform,
    nystrom_fit_transform,
    build_and_project_test_features,
)
from src.gram import (
    mixed_string_kernel,
    normalize_gram,
    generate_mkl_weights,
)

# Fixtures from conftest.py: sample_sequences, custom_alphabet


# --- 1. UNIT TESTS: COMBINED FEATURE MATRIX ---

def test_combined_features_shape(sample_sequences):
    """Verify X_combined has correct shape (N, sum of per-k vocab sizes)."""
    k_max = 3
    mismatches = 1
    weights = [0.0, 0.4, 0.6]  # k=1 skipped (weight=0), k=2 and k=3 active

    X_combined, per_k_vocabs = build_combined_feature_matrix(
        sample_sequences, k_max, mismatches, weights, n_jobs=1
    )

    assert X_combined.shape[0] == len(sample_sequences)
    assert sp.issparse(X_combined)

    # Total columns = sum of vocab sizes for active k values
    expected_cols = sum(len(v) for v in per_k_vocabs.values())
    assert X_combined.shape[1] == expected_cols


def test_combined_features_mkl_weighting(sample_sequences):
    """Verify that MKL weights scale the feature blocks correctly."""
    weights = [1.0, 0.0]  # Only k=1 active
    X_combined, per_k_vocabs = build_combined_feature_matrix(
        sample_sequences, k_max=2, mismatches=0, mkl_weights=weights, n_jobs=1
    )

    # With only k=1, columns should equal the k=1 vocabulary size
    assert 1 in per_k_vocabs
    assert 2 not in per_k_vocabs
    assert X_combined.shape[1] == len(per_k_vocabs[1])


def test_combined_features_inner_product_equals_gram(sample_sequences):
    """Verify X_combined · X_combined^T matches mixed_string_kernel output."""
    k_max = 3
    mismatches = 0
    weights = [0.3, 0.3, 0.4]

    X_combined, _ = build_combined_feature_matrix(
        sample_sequences, k_max, mismatches, weights, n_jobs=1
    )
    K_from_features = (X_combined @ X_combined.T).toarray()

    K_from_gram, _ = mixed_string_kernel(
        sample_sequences, k_max, m=mismatches, weights=weights, n_jobs=1
    )

    assert np.allclose(K_from_features, K_from_gram, atol=1e-10)


# --- 2. UNIT TESTS: ROW NORMALIZATION ---

def test_row_normalization_unit_norms(sample_sequences):
    """All rows of X_norm should have unit L2 norm."""
    X_combined, _ = build_combined_feature_matrix(
        sample_sequences, k_max=3, mismatches=0,
        mkl_weights=[0.3, 0.3, 0.4], n_jobs=1
    )
    X_norm, norms = normalize_rows(X_combined)

    row_norms = sp.linalg.norm(X_norm, axis=1)
    assert np.allclose(row_norms, 1.0, atol=1e-10)
    assert norms.shape == (len(sample_sequences),)
    assert np.all(norms > 0)


def test_row_normalization_matches_gram_normalization(sample_sequences):
    """Normalized features inner product should match normalize_gram output."""
    weights = [0.3, 0.3, 0.4]
    X_combined, _ = build_combined_feature_matrix(
        sample_sequences, k_max=3, mismatches=0, mkl_weights=weights, n_jobs=1
    )
    X_norm, _ = normalize_rows(X_combined)

    K_norm_from_features = (X_norm @ X_norm.T).toarray()

    K_gram, _ = mixed_string_kernel(
        sample_sequences, k_max=3, m=0, weights=weights, n_jobs=1
    )
    K_norm_from_gram = normalize_gram(K_gram)

    assert np.allclose(K_norm_from_features, K_norm_from_gram, atol=1e-10)


# --- 3. INTEGRATION TESTS: NYSTRÖM APPROXIMATION ---

def test_nystrom_feature_map_shape(sample_sequences):
    """Φ should have shape (N, n_components)."""
    weights = [0.3, 0.3, 0.4]
    n_components = 2  # Small for test (must be < N=4)

    X_combined, per_k_vocabs = build_combined_feature_matrix(
        sample_sequences, k_max=3, mismatches=0, mkl_weights=weights, n_jobs=1
    )
    X_norm, _ = normalize_rows(X_combined)

    Phi, state = nystrom_fit_transform(
        X_norm, n_components, seed=42,
        per_k_vocabs=per_k_vocabs, mkl_weights=weights,
        max_k=3, mismatches=0, n_jobs=1
    )

    assert Phi.shape == (len(sample_sequences), n_components)
    assert isinstance(state, NystromState)
    assert state.n_components == n_components
    assert state.W_inv_sqrt.shape == (n_components, n_components)
    assert len(state.landmark_indices) == n_components


def test_nystrom_approximation_quality(sample_sequences):
    """Φ·Φ^T should approximate the normalized Gram matrix."""
    weights = [0.3, 0.3, 0.4]
    n_components = 3  # Use most of the 4 samples for good approximation

    X_combined, per_k_vocabs = build_combined_feature_matrix(
        sample_sequences, k_max=3, mismatches=0, mkl_weights=weights, n_jobs=1
    )
    X_norm, _ = normalize_rows(X_combined)

    Phi, _ = nystrom_fit_transform(
        X_norm, n_components, seed=42,
        per_k_vocabs=per_k_vocabs, mkl_weights=weights,
        max_k=3, mismatches=0, n_jobs=1
    )

    K_approx = Phi @ Phi.T

    K_gram, _ = mixed_string_kernel(
        sample_sequences, k_max=3, m=0, weights=weights, n_jobs=1
    )
    K_exact = normalize_gram(K_gram)

    # Relative Frobenius error should be small
    error = np.linalg.norm(K_approx - K_exact, 'fro')
    baseline = np.linalg.norm(K_exact, 'fro')
    relative_error = error / baseline if baseline > 0 else error

    # With only N=4 samples and m=3 landmarks, the approximation is coarse.
    # Real-world data (N=100k, m=2000) will be much tighter.
    assert relative_error < 0.5, (
        f"Nyström approximation too poor: relative error = {relative_error:.4f}"
    )


def test_nystrom_clamps_n_components():
    """If n_components >= N, it should be clamped without error."""
    # Create tiny data: 3 sequences
    sequences = ["ATGC", "GCTA", "AAAA"]
    weights = [1.0]

    X_combined, per_k_vocabs = build_combined_feature_matrix(
        sequences, k_max=1, mismatches=0, mkl_weights=weights, n_jobs=1
    )
    X_norm, _ = normalize_rows(X_combined)

    # n_components=10 > N=3 → should clamp to 2
    Phi, state = nystrom_fit_transform(
        X_norm, n_components=10, seed=42,
        per_k_vocabs=per_k_vocabs, mkl_weights=weights,
        max_k=1, mismatches=0, n_jobs=1
    )

    assert state.n_components == 2  # N - 1
    assert Phi.shape == (3, 2)


# --- 4. INTEGRATION TESTS: OCSVM WITH NYSTRÖM FEATURES ---

def test_ocsvm_linear_accepts_nystrom_features(sample_sequences):
    """OneClassSVM(kernel='linear') should train and predict on Φ."""
    weights = [0.3, 0.3, 0.4]
    n_components = 3

    X_combined, per_k_vocabs = build_combined_feature_matrix(
        sample_sequences, k_max=3, mismatches=0, mkl_weights=weights, n_jobs=1
    )
    X_norm, _ = normalize_rows(X_combined)

    Phi, _ = nystrom_fit_transform(
        X_norm, n_components, seed=42,
        per_k_vocabs=per_k_vocabs, mkl_weights=weights,
        max_k=3, mismatches=0, n_jobs=1
    )

    svm = OneClassSVM(kernel='linear', nu=0.5)
    svm.fit(Phi)

    predictions = svm.predict(Phi)
    scores = svm.decision_function(Phi)

    assert predictions.shape == (len(sample_sequences),)
    assert scores.shape == (len(sample_sequences),)
    assert set(predictions).issubset({-1, 1})


def test_nystrom_scores_correlate_with_exact(sample_sequences):
    """Decision scores from Nyström and exact backends should be highly correlated."""
    weights = [0.3, 0.3, 0.4]
    nu = 0.5
    n_components = 3

    # --- Exact (precomputed) path ---
    K_gram, _ = mixed_string_kernel(
        sample_sequences, k_max=3, m=0, weights=weights, n_jobs=1
    )
    K_norm = normalize_gram(K_gram)

    svm_exact = OneClassSVM(kernel='precomputed', nu=nu)
    svm_exact.fit(K_norm)
    scores_exact = svm_exact.decision_function(K_norm)

    # --- Nyström path ---
    X_combined, per_k_vocabs = build_combined_feature_matrix(
        sample_sequences, k_max=3, mismatches=0, mkl_weights=weights, n_jobs=1
    )
    X_norm, _ = normalize_rows(X_combined)
    Phi, _ = nystrom_fit_transform(
        X_norm, n_components, seed=42,
        per_k_vocabs=per_k_vocabs, mkl_weights=weights,
        max_k=3, mismatches=0, n_jobs=1
    )

    svm_nystrom = OneClassSVM(kernel='linear', nu=nu)
    svm_nystrom.fit(Phi)
    scores_nystrom = svm_nystrom.decision_function(Phi)

    # Spearman correlation — with only 4 samples the ranking can vary;
    # real-world data with N >> m will show ρ > 0.9.
    rho, _ = spearmanr(scores_exact, scores_nystrom)
    assert rho > 0.4, (
        f"Score ranking correlation too low: Spearman ρ = {rho:.4f}"
    )


# --- 5. INTEGRATION TESTS: TEST PROJECTION ---

def test_test_features_match_train_vocab(sample_sequences):
    """Test features should use training vocabulary and have matching columns."""
    train_seqs = sample_sequences[:2]
    test_seqs = sample_sequences[2:]
    weights = [0.3, 0.3, 0.4]

    X_train, per_k_vocabs = build_combined_feature_matrix(
        train_seqs, k_max=3, mismatches=0, mkl_weights=weights, n_jobs=1
    )

    X_test = build_combined_test_features(
        test_seqs, per_k_vocabs, max_k=3, mismatches=0,
        mkl_weights=weights, n_jobs=1
    )

    # Column count must match
    assert X_test.shape[1] == X_train.shape[1]
    assert X_test.shape[0] == len(test_seqs)


def test_nystrom_test_projection_shape(sample_sequences):
    """build_and_project_test_features should produce correct shape."""
    train_seqs = sample_sequences[:3]
    test_seqs = sample_sequences[3:]
    weights = [0.3, 0.3, 0.4]
    n_components = 2

    X_combined, per_k_vocabs = build_combined_feature_matrix(
        train_seqs, k_max=3, mismatches=0, mkl_weights=weights, n_jobs=1
    )
    X_norm, _ = normalize_rows(X_combined)
    _, state = nystrom_fit_transform(
        X_norm, n_components, seed=42,
        per_k_vocabs=per_k_vocabs, mkl_weights=weights,
        max_k=3, mismatches=0, n_jobs=1
    )

    Phi_test = build_and_project_test_features(test_seqs, state, n_jobs=1)

    assert Phi_test.shape == (len(test_seqs), n_components)


# --- 6. UNIT TESTS: NUMERICAL STABILITY ---

def test_eigenvalue_truncation():
    """Near-singular W should be handled without NaN/Inf."""
    # Create features where some dimensions are nearly collinear
    sequences = ["AAAA", "AAAC", "AAAT"]  # Very similar sequences
    weights = [1.0]

    X_combined, per_k_vocabs = build_combined_feature_matrix(
        sequences, k_max=1, mismatches=0, mkl_weights=weights, n_jobs=1
    )
    X_norm, _ = normalize_rows(X_combined)

    Phi, state = nystrom_fit_transform(
        X_norm, n_components=2, seed=42,
        per_k_vocabs=per_k_vocabs, mkl_weights=weights,
        max_k=1, mismatches=0, n_jobs=1
    )

    assert not np.any(np.isnan(Phi)), "NaN detected in Nyström features"
    assert not np.any(np.isinf(Phi)), "Inf detected in Nyström features"


# --- 7. SERIALIZATION ROUND-TRIP ---

def test_nystrom_state_serialization(sample_sequences, tmp_path):
    """NystromState should survive a joblib save/load round-trip."""
    import dataclasses
    import joblib

    weights = [0.3, 0.3, 0.4]
    X_combined, per_k_vocabs = build_combined_feature_matrix(
        sample_sequences, k_max=3, mismatches=0, mkl_weights=weights, n_jobs=1
    )
    X_norm, _ = normalize_rows(X_combined)
    _, state = nystrom_fit_transform(
        X_norm, n_components=2, seed=42,
        per_k_vocabs=per_k_vocabs, mkl_weights=weights,
        max_k=3, mismatches=0, n_jobs=1
    )

    # Save as dict (same as ModelArtifact serialization)
    save_path = tmp_path / "test_state.pkl"
    state_dict = dataclasses.asdict(state)
    joblib.dump(state_dict, save_path)

    # Load and reconstruct
    loaded_dict = joblib.load(save_path)
    loaded_state = NystromState(**loaded_dict)

    assert loaded_state.n_components == state.n_components
    assert loaded_state.max_k == state.max_k
    assert loaded_state.mismatches == state.mismatches
    assert np.array_equal(loaded_state.landmark_indices, state.landmark_indices)
    assert np.allclose(loaded_state.W_inv_sqrt, state.W_inv_sqrt)
    assert (loaded_state.X_landmarks_norm - state.X_landmarks_norm).nnz == 0

    # Verify projection produces same results after round-trip
    Phi_original = nystrom_transform(X_norm, state)
    Phi_loaded = nystrom_transform(X_norm, loaded_state)
    assert np.allclose(Phi_original, Phi_loaded, atol=1e-12)
