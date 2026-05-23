"""
train_and_save_svm.py
Strictly executes the training pipeline on healthy baseline data
and saves the trained precomputed kernel SVM for future inference.
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
from src.kernels import generate_mkl_weights, mixed_string_kernel, normalize_gram
from src.model_io import save_svm_model
from experiments.experiments_utils import (
    setup_logger,
    create_base_parser,
    add_data_dir_arg,
    add_cache_dir_arg,
    add_train_sampling_arg,
    add_seed_arg,
    add_kernel_args,
    add_nu_arg,
    add_seq_fpr_arg,
    add_execution_args,
    build_train_normal_files,
)

logger = setup_logger(__name__)

def main():
    parser = create_base_parser("Run Sequence Novelty Detection Experiments")
    add_data_dir_arg(parser, required=True)
    add_cache_dir_arg(parser, project_root)
    add_train_sampling_arg(parser)
    add_seed_arg(parser)
    add_kernel_args(parser)
    add_nu_arg(parser)
    add_seq_fpr_arg(parser)
    add_execution_args(parser)
    parser.add_argument("--model-name", type=str, default="ocsvm_pretrained.pkl", help="Name of the saved model file")
    args = parser.parse_args()
    logger.info("=====================================================")
    logger.info(" COLON CANCER SOMATIC DETECTION: PURE MODEL TRAINING")
    logger.info("=====================================================")
    
    # 1. Define ONLY the Training Split (Healthy Baseline)
    train_normal_files = build_train_normal_files(args.data_dir)
    
    # 2. Load Data (Pass empty lists for the test sets)
    logger.info("Loading training cohort...")
    train_data, _, _, _ = load_tracked_patient_cohort(
        train_normal_files, [], [], args.max_train, 0, 0, args.seed, args.cache_dir, logger
    )

    start_time = time.perf_counter()
    
    # 3. Kernel Computation (Train vs Train ONLY)
    mkl_weights = generate_mkl_weights(args.max_k, noise_threshold=max(1, 2 * args.mismatches))
    logger.info(f"\nComputing Explicit Sparse Mismatch Kernel (Train x Train)...")
    
    K_train, train_states = mixed_string_kernel(
        sequences=train_data, 
        k_max=args.max_k, 
        m=args.mismatches, 
        weights=mkl_weights, 
        n_jobs=args.n_jobs  
    )
    K_train = normalize_gram(K_train)
    
    # 4. Fit the Model
    logger.info(f"\nFitting One-Class SVM (nu={args.nu_param})...")
    svm = OneClassSVM(kernel='precomputed', nu=args.nu_param)
    svm.fit(K_train)
    
    # 4.5 Compute tau_seq
    tau_percentile = 100.0 * (1.0 - args.seq_fpr)
    logger.info(f"Computing tau_seq ({args.seq_fpr*100:.2f}% FPR) from training data...")
    train_scores = -svm.decision_function(K_train)
    tau_seq = float(np.percentile(train_scores, tau_percentile))
    logger.info(f"tau_seq calibrated to: {tau_seq:.4f}")
    
    # 5. Save the Artifact
    save_dir = os.path.join(project_root, "models")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, args.model_name)

    save_svm_model(
        svm=svm,
        train_sequences=train_data,
        max_k=args.max_k,
        mismatches=args.mismatches,
        nu_param=args.nu_param,
        mkl_weights=mkl_weights,
        save_path=save_path,
        train_states=train_states,
        tau_seq=tau_seq,
    )

    elapsed = time.perf_counter() - start_time
    logger.info(f"Training time: {elapsed:.2f} seconds")
    logger.info("Training complete. Model safely stored and ready for inference.")

if __name__ == "__main__":
    main()