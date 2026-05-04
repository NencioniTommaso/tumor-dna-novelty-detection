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
    parser.add_argument("--cache-dir", type=str, default=os.path.join(project_root, "data", ".fai_cache"),
                        help="Path for the fasta index cache (default: data/.fai_cache).")

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

def generate_mkl_weights(max_k: int, m: int = 0, scaling: str = 'linear') -> list[float]:
    """
    Dynamically generates an array of Multiple Kernel Learning (MKL) weights.
    The noise threshold automatically scales with `m` to prevent short k-mers 
    from dominating the gram matrix with false-positive mismatch alignments.
    """
    # Dynamic threshold: e.g., if m=0 -> 1, if m=1 -> 2, if m=2 -> 4
    noise_threshold = max(1, 2 * m) 
    
    weights = []
    for k in range(1, max_k + 1):
        if k <= noise_threshold:
            weights.append(0.0)
        else:
            # Shift the base so the first valid k-mer starts at a weight > 0
            base_val = k - noise_threshold
            if scaling == 'linear':
                weights.append(float(base_val))
            elif scaling == 'quadratic':
                weights.append(float(base_val ** 2))
                
    total = sum(weights)
    
    # Safety check in case the threshold silences all k-mers
    if total == 0:
        return [0.0] * max_k
        
    return [round(w / total, 4) for w in weights]