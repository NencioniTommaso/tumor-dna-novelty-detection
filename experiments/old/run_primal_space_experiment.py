"""
run_primal_space_experiment.py
Executes a highly scalable, Data-Parallel ML pipeline for Colon Cancer Novelty Detection.
Fully saturates CPU cores by chunking datasets with fixed biological vocabularies.
"""

import time
import sys
import os
import logging
import itertools

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDOneClassSVM
from sklearn.metrics import classification_report, roc_auc_score
from joblib import Parallel, delayed

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.data_utils import load_patient_cohort
from src.kernels import extract_features, generate_mkl_weights
from experiments.experiments_utils import (
    setup_logger,
    create_base_parser,
    add_data_dir_arg,
    add_cache_dir_arg,
    add_sampling_args,
    add_seed_arg,
    add_kernel_args,
    add_nu_arg,
    add_execution_args,
)

logger = setup_logger(__name__)

def generate_dna_vocab(k: int) -> dict:
    """Generates the absolute biological vocabulary mapping for a given k."""
    return {"".join(p): i for i, p in enumerate(itertools.product('ACGT', repeat=k))}

def _extract_scaled_chunk(chunk: list[str], k: int, mismatches: int, weight: float, vocab: dict) -> sp.csr_matrix:
    """Worker function: Extracts features for a data chunk using a fixed vocabulary."""
    X_k = extract_features(chunk, k=k, m=mismatches, vocabulary=vocab)
    return X_k.multiply(np.sqrt(weight))

def evaluate_primal_detector(X_train: sp.csr_matrix, X_test: sp.csr_matrix, y_test_true: np.ndarray, nu: float, seed: int):
    logger.info(f"Fitting SGDOneClassSVM (nu={nu}) on {X_train.shape[0]} samples with {X_train.shape[1]} sparse features...")
    logger.info("Using Stochastic Gradient Descent for O(N) linear time complexity.")
    
    # We use a high max_iter and strict tolerance to force SGD to find the optimal boundary
    svm = SGDOneClassSVM(nu=nu, random_state=seed, max_iter=1000, tol=1e-3)
    
    svm.fit(X_train)
    
    logger.info("Generating predictions and computing anomaly scores...")
    predictions = svm.predict(X_test)
    anomaly_scores = svm.decision_function(X_test)
    
    # Invert scores for ROC-AUC (negative scores indicate anomalies)
    auc = roc_auc_score(y_test_true == -1, -anomaly_scores)
    report_str = classification_report(
        y_test_true, 
        predictions, 
        target_names=['Cancer (-1)', 'Healthy (1)'],
        zero_division=0
    )
    
    return auc, report_str

def main():
    parser = create_base_parser("Run Sequence Novelty Detection Experiments")
    add_data_dir_arg(parser, required=True)
    add_cache_dir_arg(parser, project_root)
    add_sampling_args(parser)
    add_seed_arg(parser)
    add_kernel_args(parser)
    add_nu_arg(parser)
    add_execution_args(parser)
    args = parser.parse_args()
    
    logger.info("=====================================================")
    logger.info(" SCALABLE SOMATIC DETECTION: DATA-PARALLEL PRIMAL SPACE")
    logger.info("=====================================================")
    
    train_normal_files = [os.path.join(args.data_dir, f"Healthy_{i}_merged_subset_1200000.fa") for i in range(2, 6)]
    test_normal_files  = [os.path.join(args.data_dir, f"Healthy_{i}_merged_subset_1200000.fa") for i in range(6, 8)]
    test_tumor_files   = [os.path.join(args.data_dir, f"Colo_{i}_merged_subset_1200000.fa") for i in range(11, 14)]
    
    train_data, test_data, y_test_true = load_patient_cohort(
        train_normal_files, test_normal_files, test_tumor_files,
        max_train=args.max_train, max_test_normal=args.max_test_normal,
        max_test_tumor=args.max_test_tumor, random_seed=args.seed,
        index_cache_dir=args.cache_dir
    )
    
    all_data = train_data + test_data
    num_train = len(train_data)
    
    # --- Data Chunking Strategy ---
    n_cores = args.n_jobs if args.n_jobs > 0 else os.cpu_count()
    # Create ~4 chunks per core to ensure smooth load balancing across threads
    chunk_size = max(1, len(all_data) // (n_cores * 4))
    chunks = [all_data[i:i + chunk_size] for i in range(0, len(all_data), chunk_size)]
    logger.info(f"Split {len(all_data)} sequences into {len(chunks)} chunks for {n_cores} CPU cores.")
    
    mkl_weights = generate_mkl_weights(args.max_k, noise_threshold=2)
    active_ks = [k for k in range(1, args.max_k + 1) if mkl_weights[k-1] > 0.0]
    
    start_time = time.time()
    
    # --- Data-Parallel Feature Extraction ---
    X_k_matrices = []
    for k in active_ks:
        vocab = generate_dna_vocab(k)
        weight = mkl_weights[k-1]
        logger.info(f"Extracting primal features for k={k} (Vocab size: {len(vocab)}) across {len(chunks)} parallel jobs...")
        
        # Parallelize the chunks instead of the Ks!
        chunk_results = Parallel(n_jobs=args.n_jobs)(
            delayed(_extract_scaled_chunk)(chunk, k, args.mismatches, weight, vocab)
            for chunk in chunks
        )
        
        # Vertically stack the chunks back into a single matrix for this K
        X_k_matrices.append(sp.vstack(chunk_results, format='csr'))
    
    logger.info("Horizontally stacking final feature blocks...")
    X_full = sp.hstack(X_k_matrices, format='csr')
    
    #logger.info("Applying L2 Normalization to primal features (Spherical Projection)...")
    #X_full = normalize(X_full, norm='l2', axis=1)

    logger.info("Applying TF-IDF Transformation (Downweights common DNA, highlights rare mutations)...")
    tfidf = TfidfTransformer(norm='l2', sublinear_tf=True)
    X_full = tfidf.fit_transform(X_full)
    
    X_train, X_test = X_full[:num_train, :], X_full[num_train:, :]
    
    auc, report_str = evaluate_primal_detector(X_train, X_test, y_test_true, args.nu_param, args.seed)
    elapsed = time.time() - start_time
    
    logger.info("=====================================================")
    logger.info(" FINAL RESULTS: SCALABLE PIPELINE")
    logger.info("=====================================================")
    logger.info(f"Total Sequences Processed : {X_full.shape[0]:,}")
    logger.info(f"Total Sparse Features     : {X_full.shape[1]:,}")
    logger.info(f"Execution Time            : {elapsed:.2f} seconds")
    logger.info(f"ROC-AUC Score             : {auc:.4f}")
    logger.info(f"\nClassification Report:\n{report_str}")

if __name__ == "__main__":
    main()