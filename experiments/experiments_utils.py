"""
experiment_utils.py
Contains shared utility functions for running machine learning experiments,
including CLI parsing, logging setup, and hyperparameter generation.
"""

import argparse
import logging
import os
import sys

def setup_logger(name: str) -> logging.Logger:
    """Configures and returns a standard logger for experiments."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=sys.stdout
    )
    return logging.getLogger(name)

def create_base_parser(description: str) -> argparse.ArgumentParser:
    """Creates a base argument parser for experiment CLIs."""
    return argparse.ArgumentParser(description=description)


def add_data_dir_arg(parser: argparse.ArgumentParser, required: bool = True) -> None:
    parser.add_argument(
        "--data-dir",
        type=str,
        required=required,
        help="Path to the directory containing the FASTA files.",
    )


def add_cache_dir_arg(parser: argparse.ArgumentParser, project_root: str) -> None:
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=os.path.join(project_root, "data", ".fai_cache"),
        help="Path for the fasta index cache (default: data/.fai_cache).",
    )


def add_kernel_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-k",
        type=int,
        default=6,
        help="Maximum k-mer size for the Mixed String Kernel (default: 6).",
    )
    parser.add_argument(
        "--mismatches",
        type=int,
        default=1,
        help="Allowed mismatch distance (default: 1).",
    )


def add_nu_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--nu-param",
        type=float,
        default=0.2,
        help="One-Class SVM nu parameter / expected anomaly rate (default: 0.2).",
    )


def add_seq_fpr_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--seq-fpr",
        type=float,
        default=0.01,
        help="Sequence-level False Positive Rate for absolute thresholding (default: 0.01).",
    )


def add_train_sampling_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-train",
        type=int,
        default=18000,
        help="Max normal sequences for training (default: 18000).",
    )


def add_test_sampling_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-test-normal",
        type=int,
        default=1500,
        help="Max healthy sequences for testing (default: 1500).",
    )
    parser.add_argument(
        "--max-test-tumor",
        type=int,
        default=1500,
        help="Max tumor sequences for testing (default: 1500).",
    )


def add_seed_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )


def add_execution_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of CPU cores to use. -1 uses all (default: -1).",
    )
    parser.add_argument(
        "--disable-kde-downsampling",
        action="store_true",
        help="Disable KDE distance downsampling (Warning: slow on large datasets).",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=None,
        help="Output directory for OA KDE plots. If not set, no plots are generated.",
    )


def add_model_path_arg(parser: argparse.ArgumentParser, project_root: str) -> None:
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.path.join(project_root, "models", "ocsvm_pretrained.pkl"),
        help="Path to the saved .pkl model artifact.",
    )


def add_patient_file_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--patient-file",
        required=True,
        help="Path to the patient's FASTA file.",
    )


def add_sample_size_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1500,
        help="Number of sequences to sample from the patient.",
    )


def build_train_normal_files(data_dir: str) -> list[str]:
    return [
        os.path.join(data_dir, f"Healthy_{i}_merged_subset_1200000.fa")
        for i in range(2, 6)
    ]


def build_test_normal_files(data_dir: str) -> list[str]:
    return [
        os.path.join(data_dir, f"Healthy_{i}_merged_subset_1200000.fa")
        for i in range(6, 8)
    ]


def build_tumor_files(data_dir: str) -> list[str]:
    return [
        os.path.join(data_dir, f"Colo_{i}_merged_subset_1200000.fa")
        for i in range(1, 11)
        if i != 9
    ]


def build_default_cohorts(data_dir: str) -> tuple[list[str], list[str], list[str]]:
    train_normal_files = build_train_normal_files(data_dir)
    test_normal_files = build_test_normal_files(data_dir)
    test_tumor_files = build_tumor_files(data_dir)
    return train_normal_files, test_normal_files, test_tumor_files


def build_validation_files(data_dir: str) -> tuple[list[str], list[str]]:
    return build_test_normal_files(data_dir), build_tumor_files(data_dir)


def validate_files_exist(file_paths: list[str], logger: logging.Logger) -> bool:
    missing_files = [path for path in file_paths if not os.path.exists(path)]
    if not missing_files:
        return True

    for path in missing_files:
        logger.error(f"Cannot find file: {path}")
    return False