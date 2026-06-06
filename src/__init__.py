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
    evaluate_patient_level_novelty,
)

# Model persistence
from src.model_io import ModelArtifact, save_svm_model, load_svm_model
