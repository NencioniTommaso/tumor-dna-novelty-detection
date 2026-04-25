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

def parse_arguments(project_root: str) -> argparse.Namespace:
    """Parses command line arguments for the experiment pipelines."""
    parser = argparse.ArgumentParser(
        description="Run Sequence Novelty Detection Experiments"
    )

    # File and Directory Paths
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to the directory containing the FASTA files.")
    parser.add_argument("--cache-dir", type=str, default=os.path.join(project_root, ".fai_cache"),
                        help="Path for the fasta index cache (default: .fai_cache inside project root).")

    # Biological/Model Hyperparameters
    parser.add_argument("--max-k", type=int, default=6,
                        help="Maximum k-mer size for the Mixed String Kernel (default: 6).")
    parser.add_argument("--mismatches", type=int, default=1,
                        help="Allowed mismatch distance (default: 1).")
    parser.add_argument("--nu-param", type=float, default=0.2,
                        help="One-Class SVM nu parameter / expected anomaly rate (default: 0.2).")

    # Data Sampling Constraints
    parser.add_argument("--max-train", type=int, default=18000,
                        help="Max normal sequences for training (default: 18000).")
    parser.add_argument("--max-test-normal", type=int, default=1500,
                        help="Max healthy sequences for testing (default: 1500).")
    parser.add_argument("--max-test-tumor", type=int, default=1500,
                        help="Max tumor sequences for testing (default: 1500).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42).")

    # Execution Parameters
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Number of CPU cores to use. -1 uses all (default: -1).")

    return parser.parse_args()

def generate_mkl_weights(max_k: int, noise_threshold: int = 2, scaling: str = 'linear') -> list[float]:
    """
    Dynamically generates an array of ascending Multiple Kernel Learning (MKL) weights.
    Silences small k-mers (noise) and rewards larger structural motifs.
    """
    weights = []
    for k in range(1, max_k + 1):
        if k <= noise_threshold:
            weights.append(0.0)
        else:
            if scaling == 'linear':
                weights.append(float(k - noise_threshold))
            elif scaling == 'quadratic':
                weights.append(float((k - noise_threshold) ** 2))
                
    total = sum(weights)
    return [round(w / total, 4) for w in weights]