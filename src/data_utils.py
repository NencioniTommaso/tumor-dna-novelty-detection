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


def load_training_cohort_tracked_indices(
    train_files: list[str],
    max_train: int,
    seed: int,
    cache_dir: str,
) -> tuple[list[str], dict[str, np.ndarray]]:
    """Load training data and record which FASTA indices were sampled per file.

    This variant of training-data loading returns both the sequences and
    a mapping from each file path to the array of sampled indices.  The
    indices are needed by the multi-round LOO experiment so that test
    sampling from the *same* files can exclude sequences already used
    for training.

    Parameters
    ----------
    train_files : list[str]
        Paths to the healthy FASTA files used for training.
    max_train : int
        Maximum total sequences to sample across all files.
    seed : int
        Random seed for reproducibility.
    cache_dir : str
        Path for the FASTA index cache.

    Returns
    -------
    train_data : list[str]
        Sampled training sequences (upper-cased).
    sampled_indices_per_file : dict[str, np.ndarray]
        ``{file_path: array_of_sampled_indices}`` for each training file.
    """
    np.random.seed(seed)

    if not train_files:
        return [], {}

    seqs_per_file = max_train // len(train_files)
    all_seqs: list[str] = []
    sampled_indices_per_file: dict[str, np.ndarray] = {}

    for file_path in train_files:
        logger.info(f"  -> Loading Train (Normal): {os.path.basename(file_path)}")
        reader = MMapFastaReader(file_path, index_cache_dir=cache_dir)
        total_available = len(reader.offsets)
        num_to_sample = min(seqs_per_file, total_available)
        sampled_indices = np.random.choice(total_available, num_to_sample, replace=False)
        raw_seqs = [reader.get_seq(i) for i in sampled_indices]
        reader.close()

        clean_seqs = [s.upper() for s in raw_seqs if s is not None]
        all_seqs.extend(clean_seqs)
        sampled_indices_per_file[file_path] = sampled_indices

    return all_seqs, sampled_indices_per_file


def sample_non_overlapping_rounds(
    fasta_path: str,
    n_rounds: int,
    seqs_per_round: int,
    seed: int,
    cache_dir: str,
    excluded_indices: np.ndarray | None = None,
) -> list[list[str]]:
    """Sample ``n_rounds`` batches of sequences with zero overlap.

    Draws ``n_rounds * seqs_per_round`` unique indices from the FASTA
    file in a single shot (excluding any indices in *excluded_indices*),
    then partitions them into ``n_rounds`` equal chunks.

    Parameters
    ----------
    fasta_path : str
        Path to the FASTA file.
    n_rounds : int
        Number of non-overlapping test rounds.
    seqs_per_round : int
        Sequences to draw per round.
    seed : int
        Random seed for reproducibility.
    cache_dir : str
        Path for the FASTA index cache.
    excluded_indices : np.ndarray or None
        Indices to exclude (e.g. those already used for training).

    Returns
    -------
    rounds : list[list[str]]
        A list of *n_rounds* lists, each containing *seqs_per_round*
        upper-cased sequences.

    Raises
    ------
    ValueError
        If there are not enough available indices to fill all rounds.
    """
    np.random.seed(seed)

    reader = MMapFastaReader(fasta_path, index_cache_dir=cache_dir)
    total_available = len(reader.offsets)

    # Build the pool of available indices
    all_indices = np.arange(total_available)
    if excluded_indices is not None and len(excluded_indices) > 0:
        excluded_set = set(excluded_indices.tolist())
        all_indices = np.array([i for i in all_indices if i not in excluded_set])

    total_needed = n_rounds * seqs_per_round
    if len(all_indices) < total_needed:
        raise ValueError(
            f"Not enough sequences in {os.path.basename(fasta_path)}: "
            f"need {total_needed} but only {len(all_indices)} available "
            f"(after excluding {total_available - len(all_indices)} training indices)."
        )

    # Single draw, then partition
    chosen = np.random.choice(all_indices, total_needed, replace=False)
    round_indices = [chosen[i * seqs_per_round:(i + 1) * seqs_per_round] for i in range(n_rounds)]

    rounds: list[list[str]] = []
    for r_idx, indices in enumerate(round_indices):
        raw_seqs = [reader.get_seq(int(i)) for i in indices]
        clean_seqs = [s.upper() for s in raw_seqs if s is not None]
        rounds.append(clean_seqs)
        logger.info(
            f"  -> Round {r_idx + 1}/{n_rounds}: sampled {len(clean_seqs)} sequences "
            f"from {os.path.basename(fasta_path)}"
        )

    reader.close()
    return rounds