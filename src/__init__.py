"""
src — Tumor DNA Novelty Detection Library

Public API re-exports for convenience.
"""

# Core data I/O
from src.fasta_reader import MMapFastaReader
from src.data_utils import (
    load_tracked_patient_cohort,
    load_train_cohort_only,
    load_test_cohort_only,
    load_training_cohort_tracked_indices,
    sample_non_overlapping_rounds,
)

# Mismatch combinatorics
from src.mismatch import (
    EPIGENETIC_ALPHABET,
    generate_mismatch_neighborhood,
    generate_weighted_mismatch_neighborhood,
    build_full_vocabulary,
)

# Feature extraction
from src.features import (
    extract_features,
    extract_features_weighted,
)

# Gram matrix computation
from src.gram import (
    generate_mkl_weights,
    ensure_mkl_weights,
    mixed_string_kernel,
    normalize_gram,
    compute_asymmetric_normalized_kernel,
)

# Evaluation
from src.evaluation import (
    evaluate_novelty_detector,
    evaluate_novelty_detector_nystrom,
    evaluate_patient_level_novelty,
)

# Model persistence
from src.model_io import ModelArtifact, save_svm_model, load_svm_model

# Nyström kernel approximation
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

