"""
train_and_save_svm_nystrom.py
Trains an OC-SVM using Nyström kernel approximation for scalable training.
Uses OneClassSVM(kernel='linear') on the Nyström feature map Φ.

The Nyström method avoids materializing the N×N Gram matrix, reducing
memory from O(N²) to O(N·m) and enabling training on 100k+ sequences.
"""

import sys
import os
import time
import numpy as np
from sklearn.svm import OneClassSVM

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.data_utils import load_tracked_patient_cohort
from src.gram import generate_mkl_weights
from src.nystrom import (
    build_combined_feature_matrix,
    normalize_rows,
    nystrom_fit_transform,
)
from src.model_io import save_svm_model, ModelArtifact
from experiments.experiments_utils import (
    setup_logger,
    create_base_parser,
    add_data_dir_arg,
    add_cache_dir_arg,
    add_train_sampling_arg,
    add_seed_arg,
    add_kernel_args,
    add_nu_arg,
    add_execution_args,
    build_train_normal_files,
)

logger = setup_logger(__name__)


def main():
    parser = create_base_parser("Train OC-SVM with Nyström kernel approximation")
    add_data_dir_arg(parser, required=True)
    add_cache_dir_arg(parser, project_root)
    add_train_sampling_arg(parser)
    add_seed_arg(parser)
    add_kernel_args(parser)
    add_nu_arg(parser)
    add_execution_args(parser)
    parser.add_argument(
        "--n-components", type=int, default=2000,
        help="Number of Nyström landmark points (default: 2000)."
    )
    parser.add_argument(
        "--model-name", type=str, default="ocsvm_nystrom.pkl",
        help="Name of the saved model file"
    )
    parser.add_argument(
        "--store-sequences", action="store_true",
        help="Store training sequences in the artifact (large at scale)."
    )
    args = parser.parse_args()

    logger.info("=====================================================")
    logger.info(" NYSTRÖM SCALABLE OC-SVM TRAINING")
    logger.info("=====================================================")
    logger.info(f"  max_k={args.max_k}  mismatches={args.mismatches}  nu={args.nu_param}")
    logger.info(f"  max_train={args.max_train}  n_components={args.n_components}")

    # 1. Load Training Data
    train_normal_files = build_train_normal_files(args.data_dir)

    logger.info("Loading training cohort...")
    train_data, _, _, _ = load_tracked_patient_cohort(
        train_normal_files, [], [], args.max_train, 0, 0, args.seed, args.cache_dir
    )

    start_time = time.perf_counter()

    # 2. Build Combined Feature Matrix
    mkl_weights = generate_mkl_weights(args.max_k, noise_threshold=max(1, 2 * args.mismatches))
    logger.info(f"\nMKL weights: {mkl_weights}")
    logger.info(f"Building combined feature matrix (k_max={args.max_k}, m={args.mismatches})...")

    X_combined, per_k_vocabs = build_combined_feature_matrix(
        train_data, args.max_k, args.mismatches, mkl_weights, args.n_jobs
    )

    # 3. Normalize and Apply Nyström
    logger.info("Normalizing features (row-wise L2 — equivalent to cosine kernel)...")
    X_norm, _ = normalize_rows(X_combined)

    logger.info(f"Fitting Nyström approximation (n_components={args.n_components})...")
    Phi_train, nystrom_state = nystrom_fit_transform(
        X_norm, args.n_components, args.seed,
        per_k_vocabs, mkl_weights, args.max_k, args.mismatches,
        n_jobs=args.n_jobs
    )

    # 4. Fit the Model
    logger.info(f"\nFitting OneClassSVM(kernel='linear', nu={args.nu_param})...")
    svm = OneClassSVM(kernel='linear', nu=args.nu_param)
    svm.fit(Phi_train)

    # 5. Save the Artifact
    save_dir = os.path.join(project_root, "models")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, args.model_name)

    artifact = ModelArtifact(
        model=svm,
        train_sequences=train_data if args.store_sequences else [],
        max_k=args.max_k,
        mismatches=args.mismatches,
        nu_param=args.nu_param,
        mkl_weights=mkl_weights,
        backend="nystrom",
        nystrom_state=nystrom_state,
    )
    save_svm_model(artifact, save_path)

    elapsed = time.perf_counter() - start_time
    logger.info(f"\nTraining time: {elapsed:.2f} seconds")
    logger.info(f"Feature map dimensions: {Phi_train.shape}")
    logger.info(f"Training complete. Nyström model saved to {save_path}")


if __name__ == "__main__":
    main()
