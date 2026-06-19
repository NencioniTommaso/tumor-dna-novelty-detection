"""
run_inference_nystrom.py
Loads a Nyström-trained OC-SVM model and runs inference on a patient FASTA file.
Projects test sequences into the Nyström feature space and computes anomaly scores.
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
    add_sample_size_arg,
    add_cache_dir_arg,
)
from src.fasta_reader import MMapFastaReader
from src.nystrom import build_and_project_test_features
from src.model_io import load_svm_model

logger = setup_logger(__name__)


def main():
    parser = create_base_parser("Run inference using a Nyström-trained OC-SVM model.")
    add_patient_file_arg(parser)
    parser.add_argument(
        "--model-path", type=str,
        default=os.path.join(project_root, "models", "ocsvm_nystrom.pkl"),
        help="Path to the saved Nyström model artifact (.pkl).",
    )
    add_sample_size_arg(parser)
    add_cache_dir_arg(parser, project_root)
    parser.add_argument(
        "--n-jobs", type=int, default=-1,
        help="Number of CPU cores for feature extraction. -1 uses all (default: -1).",
    )
    parser.add_argument(
        "--output-csv", type=str, default=None,
        help="Optional path to save all anomaly scores as a CSV file.",
    )

    args = parser.parse_args()

    start_time = time.perf_counter()

    # 1. Load Model
    artifact = load_svm_model(args.model_path)

    if artifact.backend != "nystrom":
        logger.error(
            f"Model was trained with backend='{artifact.backend}', not 'nystrom'. "
            "Use run_inference.py for precomputed models."
        )
        sys.exit(1)

    logger.info(
        f"Loaded Nyström OC-SVM "
        f"(n_components={artifact.nystrom_state.n_components}, "
        f"k_max={artifact.max_k}, m={artifact.mismatches}, nu={artifact.nu_param})"
    )

    # 2. Load Patient Data
    logger.info(f"Loading patient data from {os.path.basename(args.patient_file)}...")
    reader = MMapFastaReader(args.patient_file, index_cache_dir=args.cache_dir)
    total_available = len(reader.offsets)
    num_to_sample = min(args.sample_size, total_available)
    sampled_indices = np.random.choice(total_available, num_to_sample, replace=False)
    raw_seqs = [reader.get_seq(i) for i in sampled_indices]
    reader.close()

    new_patient_sequences = [seq.upper() for seq in raw_seqs if seq is not None]
    logger.info(f"Projecting {len(new_patient_sequences)} sequences into Nyström feature space...")

    # 3. Project Test Sequences
    Phi_test = build_and_project_test_features(
        new_patient_sequences, artifact.nystrom_state, n_jobs=args.n_jobs
    )

    # 4. Predict
    logger.info("Predicting sequence anomalies...")
    anomaly_scores = artifact.model.decision_function(Phi_test)

    # Invert so higher = more anomalous
    inverted_scores = -anomaly_scores
    patient_score = float(np.mean(inverted_scores))

    elapsed = time.perf_counter() - start_time
    logger.info(f"Inference time: {elapsed:.2f} seconds")

    # 5. Save to CSV if requested
    if args.output_csv:
        import csv
        logger.info(f"Saving {len(inverted_scores)} anomaly scores to {args.output_csv}...")
        # Ensure directory exists
        out_dir = os.path.dirname(os.path.abspath(args.output_csv))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            
        with open(args.output_csv, 'w', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(["anomaly_score"])
            for score in inverted_scores:
                writer.writerow([score])

    logger.info("\n=====================================================")
    logger.info(f" PATIENT FASTA: {os.path.basename(args.patient_file)}")
    logger.info(f" MEAN ANOMALY SCORE: {patient_score:.4f}")
    logger.info(f" NUM SEQUENCES SCORED: {len(anomaly_scores)}")
    logger.info("=====================================================\n")


if __name__ == "__main__":
    main()
