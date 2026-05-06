"""
run_inference.py
Loads a pre-trained Patient-Level MIL model and efficiently calculates
only the asymmetric inference kernel for new patients, avoiding memory bloat.
"""

import argparse
import os
import sys
import time

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from experiments.experiments_utils import setup_logger
from src.data_utils import MMapFastaReader
from src.kernels import compute_asymmetric_normalized_kernel, generate_mkl_weights
from src.model_io import load_svm_model

logger = setup_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run inference using a saved OC-SVM model.")
    parser.add_argument("--patient-file", required=True, help="Path to the patient's FASTA file.")
    parser.add_argument(
        "--model-path",
        default=os.path.join(project_root, "models", "ocsvm_pretrained.pkl"),
        help="Path to the saved .pkl model.",
    )
    parser.add_argument("--sample-size", type=int, default=1500, help="Number of sequences to sample from the patient.")
    parser.add_argument(
        "--cache-dir",
        default=os.path.join(project_root, "data", ".fai_cache"),
        help="Path for the fasta index cache.",
    )

    args = parser.parse_args()

    start_time = time.perf_counter()

    svm, train_sequences, max_k, mismatches, mkl_weights = load_svm_model(args.model_path, logger)
    logger.info(f"Loaded SVM trained on {len(train_sequences)} sequences.")

    logger.info(f"Loading patient data from {os.path.basename(args.patient_file)}...")
    reader = MMapFastaReader(args.patient_file, index_cache_dir=args.cache_dir)
    total_available = len(reader.offsets)
    num_to_sample = min(args.sample_size, total_available)
    sampled_indices = np.random.choice(total_available, num_to_sample, replace=False)
    raw_seqs = [reader.get_seq(i) for i in sampled_indices]
    reader.close()

    new_patient_sequences = [sequence.upper() for sequence in raw_seqs if sequence is not None]
    logger.info(f"Computing asymmetric inference kernel for {len(new_patient_sequences)} sequences...")

    # Backward compatibility for models saved before MKL weights were serialized.
    if mkl_weights is None:
        logger.info("No saved MKL weights found in model artifact; recomputing for compatibility.")
        mkl_weights = generate_mkl_weights(max_k, noise_threshold=max(1, 2 * mismatches))

    K_test = compute_asymmetric_normalized_kernel(
        test_seqs=new_patient_sequences,
        train_seqs=train_sequences,
        max_k=max_k,
        mismatches=mismatches,
        mkl_weights=mkl_weights,
    )

    logger.info("Predicting sequence anomalies...")
    anomaly_scores = svm.decision_function(K_test)

    inverted_scores = -anomaly_scores
    top_k = max(1, int(len(new_patient_sequences) * 0.05))
    patient_score = float(np.mean(np.sort(inverted_scores)[-top_k:]))

    elapsed = time.perf_counter() - start_time
    logger.info(f"Inference time: {elapsed:.2f} seconds")

    logger.info("\n=====================================================")
    logger.info(f" PATIENT FASTA: {os.path.basename(args.patient_file)}")
    logger.info(f" FINAL ANOMALY SCORE: {patient_score:.4f}")
    logger.info("=====================================================\n")


if __name__ == "__main__":
    main()
