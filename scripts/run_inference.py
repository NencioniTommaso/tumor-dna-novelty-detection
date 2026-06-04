"""
run_inference.py
Loads a pre-trained Patient-Level MIL model and efficiently calculates
only the asymmetric inference kernel for new patients, avoiding memory bloat.
Outputs the raw anomaly scores for inspection.
"""

import os
import sys
import time

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from experiments.experiments_utils import (
    setup_logger,
    create_base_parser,
    add_patient_file_arg,
    add_model_path_arg,
    add_sample_size_arg,
    add_cache_dir_arg,
)
from src.data_utils import MMapFastaReader
from src.kernels import compute_asymmetric_normalized_kernel, ensure_mkl_weights
from src.model_io import load_svm_model

logger = setup_logger(__name__)


def main():
    parser = create_base_parser("Run inference using a saved OC-SVM model.")
    add_patient_file_arg(parser)
    add_model_path_arg(parser, project_root)
    add_sample_size_arg(parser)
    add_cache_dir_arg(parser, project_root)

    args = parser.parse_args()

    start_time = time.perf_counter()

    svm, train_sequences, max_k, mismatches, mkl_weights, train_states = load_svm_model(
        args.model_path
    )
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
    mkl_weights = ensure_mkl_weights(max_k, mismatches, mkl_weights)

    K_test = compute_asymmetric_normalized_kernel(
        test_seqs=new_patient_sequences,
        train_states=train_states,
        max_k=max_k,
        mismatches=mismatches,
        mkl_weights=mkl_weights,
    )

    logger.info("Predicting sequence anomalies...")
    anomaly_scores = svm.decision_function(K_test)

    # Invert so higher = more anomalous
    inverted_scores = -anomaly_scores
    patient_score = float(np.mean(inverted_scores))

    elapsed = time.perf_counter() - start_time
    logger.info(f"Inference time: {elapsed:.2f} seconds")

    logger.info("\n=====================================================")
    logger.info(f" PATIENT FASTA: {os.path.basename(args.patient_file)}")
    logger.info(f" MEAN ANOMALY SCORE: {patient_score:.4f}")
    logger.info(f" NUM SEQUENCES SCORED: {len(anomaly_scores)}")
    logger.info("=====================================================\n")


if __name__ == "__main__":
    main()
