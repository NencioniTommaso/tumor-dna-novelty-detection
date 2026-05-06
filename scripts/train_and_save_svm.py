"""
train_and_save_svm.py
Strictly executes the training pipeline on healthy baseline data
and saves the trained precomputed kernel SVM for future inference.
"""

import sys
import os
import time
from sklearn.svm import OneClassSVM

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.data_utils import load_tracked_patient_cohort
from src.kernels import generate_mkl_weights, mixed_string_kernel, normalize_gram
from src.model_io import save_svm_model
from experiments.experiments_utils import setup_logger, parse_arguments

logger = setup_logger(__name__)

def main():
    args = parse_arguments(project_root)
    logger.info("=====================================================")
    logger.info(" COLON CANCER SOMATIC DETECTION: PURE MODEL TRAINING")
    logger.info("=====================================================")
    
    # 1. Define ONLY the Training Split (Healthy Baseline)
    train_normal_files = [os.path.join(args.data_dir, f"Healthy_{i}_merged_subset_1200000.fa") for i in range(2, 6)]
    
    # 2. Load Data (Pass empty lists for the test sets)
    logger.info("Loading training cohort...")
    train_data, _, _, _ = load_tracked_patient_cohort(
        train_normal_files, [], [], args, logger
    )

    start_time = time.perf_counter()
    
    # 3. Kernel Computation (Train vs Train ONLY)
    mkl_weights = generate_mkl_weights(args.max_k, noise_threshold=max(1, 2 * args.mismatches))
    logger.info(f"\nComputing Explicit Sparse Mismatch Kernel (Train x Train)...")
    
    K_train, _ = mixed_string_kernel(
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
    
    # 5. Save the Artifact
    save_dir = os.path.join(project_root, "models")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "ocsvm_pretrained.pkl")

    save_svm_model(
        svm=svm,
        train_sequences=train_data,
        max_k=args.max_k,
        mismatches=args.mismatches,
        nu_param=args.nu_param,
        mkl_weights=mkl_weights,
        save_path=save_path,
        logger=logger,
    )

    elapsed = time.perf_counter() - start_time
    logger.info(f"Training time: {elapsed:.2f} seconds")
    logger.info("Training complete. Model safely stored and ready for inference.")

if __name__ == "__main__":
    main()