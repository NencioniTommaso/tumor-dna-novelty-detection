"""
data_utils.py
Handles cohort loading for DNA sequence anomaly detection.
Uses MMapFastaReader from fasta_reader.py for I/O.
"""

import os
import logging

import numpy as np

from src.fasta_reader import MMapFastaReader

# Configure the module-level logger
logger = logging.getLogger(__name__)


def _read_and_track(file_list, desc, max_total_seqs, label, cache_dir):
    """Read FASTA files and optionally track per-file metadata.

    This is a module-level helper so it can be reused by the decoupled
    loading functions without duplicating logic.
    """
    if not file_list:
        return [], []

    seqs_per_file = max_total_seqs // len(file_list)
    all_seqs = []
    files_info = []

    for file_path in file_list:
        logger.info(f"  -> Loading {desc}: {os.path.basename(file_path)}")
        reader = MMapFastaReader(file_path, index_cache_dir=cache_dir)
        total_available = len(reader.offsets)
        num_to_sample = min(seqs_per_file, total_available)
        sampled_indices = np.random.choice(total_available, num_to_sample, replace=False)
        raw_seqs = [reader.get_seq(i) for i in sampled_indices]
        reader.close()

        clean_seqs = [s.upper() for s in raw_seqs if s is not None]
        all_seqs.extend(clean_seqs)

        if label is not None:
            files_info.append({
                'filename': os.path.basename(file_path),
                'label': label,
                'num_sequences': len(clean_seqs)
            })

    return all_seqs, files_info


def load_tracked_patient_cohort(train_normal_files, test_normal_files, test_tumor_files, max_train: int, max_test_normal: int, max_test_tumor: int, seed: int, cache_dir: str):
    np.random.seed(seed)

    logger.info("--- Loading Training Data (Healthy Baseline) ---")
    train_data, _ = _read_and_track(train_normal_files, "Train (Normal)", max_train, None, cache_dir)

    logger.info("\n--- Loading Testing Data (Tracked Instances) ---")
    test_normal_data, normal_info = _read_and_track(test_normal_files, "Test (Normal)", max_test_normal, 1, cache_dir)
    test_tumor_data, tumor_info = _read_and_track(test_tumor_files, "Test (Tumor)", max_test_tumor, -1, cache_dir)

    test_data = test_normal_data + test_tumor_data
    test_files_info = normal_info + tumor_info
    y_test_true_seq = np.array([1] * len(test_normal_data) + [-1] * len(test_tumor_data))

    return train_data, test_data, y_test_true_seq, test_files_info


def load_train_cohort_only(train_normal_files, max_train: int, seed: int, cache_dir: str):
    """Load ONLY training data with a dedicated seed.

    Used by build_reference.py to build a reference distribution
    independently of any test data sampling.
    """
    np.random.seed(seed)
    logger.info("--- Loading Training Data (Healthy Baseline) ---")
    train_data, _ = _read_and_track(train_normal_files, "Train (Normal)", max_train, None, cache_dir)
    return train_data


def load_test_cohort_only(test_normal_files, test_tumor_files, max_test_normal: int, max_test_tumor: int, seed: int, cache_dir: str):
    """Load ONLY test data with a dedicated seed.

    Fully independent of training sampling — the RNG state
    is seeded fresh so the result does not depend on how many
    training sequences were drawn.
    """
    np.random.seed(seed)
    logger.info("\n--- Loading Testing Data (Tracked Instances) ---")
    test_normal_data, normal_info = _read_and_track(test_normal_files, "Test (Normal)", max_test_normal, 1, cache_dir)
    test_tumor_data, tumor_info = _read_and_track(test_tumor_files, "Test (Tumor)", max_test_tumor, -1, cache_dir)

    test_data = test_normal_data + test_tumor_data
    test_files_info = normal_info + tumor_info
    y_test_true_seq = np.array([1] * len(test_normal_data) + [-1] * len(test_tumor_data))

    return test_data, y_test_true_seq, test_files_info