"""
run_full_loo_exact_cached.py
Leave-One-Out Cross Validation with Exact Kernel (No Nyström Approximation).
This optimized version caches the sparse feature matrices for the 7 healthy patients
upfront to avoid redundant extraction inside the fold loop.
"""

import csv
import gc
import os
import sys
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import numpy as np
import scipy.sparse as sp
from sklearn.svm import OneClassSVM

from src.features import normalize_rows
from src.gram import generate_mkl_weights, parallel_gram_matrix, parallel_asymmetric_gram_matrix
from src.nystrom import (
    build_combined_test_features,
    normalize_rows,
)
from src.mismatch import build_full_vocabulary
from src.model_io import save_svm_model, ModelArtifact
from experiments.experiments_utils import (
    setup_logger,
    create_base_parser,
    add_data_dir_arg,
    add_cache_dir_arg,
    add_kernel_args,
    add_nu_arg,
    add_execution_args,
    build_all_healthy_files,
    validate_files_exist,
)

logger = setup_logger(__name__)

# Indexes of Colo patients to evaluate on
COLO_START = 11
COLO_END = 15

def main():
    parser = create_base_parser("Full LOO Experiment with Exact Kernel (Cached)")
    add_data_dir_arg(parser, required=True)
    add_cache_dir_arg(parser, project_root)
    add_kernel_args(parser)
    add_nu_arg(parser)
    add_execution_args(parser)
    parser.add_argument("--max-train", type=int, default=30_000,
                        help="Total training sequences across all training patients (default: 30000).")
    parser.add_argument("--max-test", type=int, default=50_000,
                        help="Sequences to infer per test subject (default: 50000).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for data loading (default: 42).")
    parser.add_argument("--test-batch-size", type=int, default=10_000,
                        help="Number of test sequences to process at once (default: 10000).")
    parser.add_argument("--start-fold", type=int, default=1,
                        help="Fold to start from (1-7). Allows resuming from a specific fold (default: 1).")
    args = parser.parse_args()

    all_healthy = build_all_healthy_files(args.data_dir)
    colo_files = [os.path.join(args.data_dir, f"Colo_{i}_merged_subset_1200000.fa") for i in range(COLO_START, COLO_END + 1)]
    
    if not validate_files_exist(all_healthy + colo_files, logger):
        sys.exit(1)

    mkl_weights = generate_mkl_weights(args.max_k, noise_threshold=max(1, 2 * args.mismatches))

    base_out_dir = os.path.join(
        project_root, "results", "full_loo_exact_cached", 
        f"m_{args.mismatches}", f"k_{args.max_k}"
    )
    models_dir = os.path.join(base_out_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    logger.info("=" * 65)
    logger.info(" FULL LOO EXPERIMENT (EXACT KERNEL - CACHED)")
    logger.info("=" * 65)
    logger.info(f"  Colo Range       : {COLO_START} to {COLO_END}")
    logger.info(f"  Max train seqs   : {args.max_train}")
    logger.info(f"  Test sequences   : {args.max_test}")
    logger.info(f"  Kernel           : max_k={args.max_k}, m={args.mismatches}")
    logger.info(f"  MKL weights      : {mkl_weights}")
    logger.info(f"  OCSVM nu         : {args.nu_param}")
    logger.info(f"  Output dir       : {base_out_dir}")
    logger.info("=" * 65)

    # ------------------------------------------------------------------
    # 0. Pre-compute Vocabulary & Cache Healthy Patient Features
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 65)
    logger.info(" PRE-CACHING TRAINING FEATURES (ONCE)")
    logger.info("=" * 65)
    
    start_k = max(1, args.mismatches + 1) if args.mismatches > 0 else 1
    active_ks = [k for k in range(start_k, args.max_k + 1) if mkl_weights[k - 1] != 0.0]
    
    if not active_ks:
        logger.error("No active k-mer sizes after MKL weight filtering.")
        sys.exit(1)
        
    per_k_vocabs = {}
    for k in active_ks:
        per_k_vocabs[k] = build_full_vocabulary(k)
        logger.info(f"Built full vocabulary for k={k} (size={len(per_k_vocabs[k])})")

    healthy_features_cache = {}
    # We distribute max_train among the 6 files we will use per fold
    seqs_per_file = args.max_train // 6 
    
    t0_cache = time.time()
    for file_path in all_healthy:
        subject_name = os.path.basename(file_path)
        logger.info(f"\n--- Extracting features for {subject_name} ---")
        
        rounds = sample_non_overlapping_rounds(
            fasta_path=file_path,
            n_rounds=1,
            seqs_per_round=seqs_per_file,
            seed=args.seed,
            cache_dir=args.cache_dir,
            excluded_indices=None,
        )
        patient_seqs = rounds[0]
        
        X_patient = build_combined_test_features(
            test_sequences=patient_seqs,
            per_k_vocabs=per_k_vocabs,
            max_k=args.max_k,
            mismatches=args.mismatches,
            mkl_weights=mkl_weights,
            n_jobs=args.n_jobs,
        )
        
        healthy_features_cache[file_path] = (patient_seqs, X_patient)
        
    logger.info(f"\nFeature extraction caching completed in {time.time() - t0_cache:.1f}s")
    
    # ------------------------------------------------------------------
    # 0b. Pre-Cache Testing Features (Tumors + All Healthy)
    # ------------------------------------------------------------------
    logger.info("\n--- Pre-caching TEST features ---")
    test_features_cache = {}
    all_test_files = all_healthy + colo_files
    test_features_dir = os.path.join(base_out_dir, "test_features")
    os.makedirs(test_features_dir, exist_ok=True)
    
    for file_path in all_test_files:
        subject_name = "_".join(os.path.basename(file_path).split("_")[:2])
        logger.info(f"Extracting test features for {subject_name} ...")
        
        rounds = sample_non_overlapping_rounds(
            fasta_path=file_path,
            n_rounds=1,
            seqs_per_round=args.max_test,
            seed=args.seed,
            cache_dir=args.cache_dir,
            excluded_indices=None,
        )
        test_seqs = rounds[0]
        
        X_test_combined = build_combined_test_features(
            test_seqs, per_k_vocabs,
            args.max_k, args.mismatches,
            mkl_weights,
            n_jobs=args.n_jobs,
        )
        X_test_norm, _ = normalize_rows(X_test_combined)
        del X_test_combined
        
        test_features_cache[file_path] = (test_seqs, X_test_norm)
        
        features_path = os.path.join(test_features_dir, f"{subject_name}_features_seed{args.seed}.npz")
        sp.save_npz(features_path, X_test_norm)
        
    logger.info(f"\nTest feature caching completed in {time.time() - t0_cache:.1f}s")
    logger.info("=" * 65)

    # ------------------------------------------------------------------
    # FOLD LOOP
    # ------------------------------------------------------------------
    summary_rows = []
    total_start = time.time()

    for held_out_id in range(args.start_fold, 8):
        fold_name = f"LOO_Healthy_{held_out_id}"
        out_dir = os.path.join(base_out_dir, fold_name)
        os.makedirs(out_dir, exist_ok=True)

        logger.info("")
        logger.info("=" * 65)
        logger.info(f" STARTING FOLD: {fold_name}")
        logger.info("=" * 65)

        held_out_file = None
        for f in all_healthy:
            if f"Healthy_{held_out_id}_" in os.path.basename(f):
                held_out_file = f
                break
        
        train_files = [f for f in all_healthy if f != held_out_file]
        test_files = [(held_out_file, "healthy")] + [(cf, "tumor") for cf in colo_files]

        logger.info(f"  Training on   : {[os.path.basename(f) for f in train_files]}")
        logger.info(f"  Held-out test : {os.path.basename(held_out_file)}")
        logger.info(f"  Tumor test    : {[os.path.basename(f) for f in colo_files]}")

        # ------------------------------------------------------------------
        # 1. & 2. Construct Combined Training Features from Cache
        # ------------------------------------------------------------------
        logger.info("\n--- Step 1 & 2: Assembling Training Features from Cache ---")
        
        train_data = []
        X_blocks = []
        for f in train_files:
            seqs, X = healthy_features_cache[f]
            train_data.extend(seqs)
            X_blocks.append(X)
            
        X_combined = sp.vstack(X_blocks, format='csr')
        N = len(train_data)
        logger.info(f"Assembled {N:,} training sequences. Matrix shape: {X_combined.shape}")
        
        t0 = time.time()
        X_norm_train, _ = normalize_rows(X_combined)
        del X_combined
        gc.collect()
        logger.info(f"Row normalization complete in {time.time() - t0:.1f}s")

        # ------------------------------------------------------------------
        # 3. Fit Linear OC-SVM -> Now Precomputed
        # ------------------------------------------------------------------
        logger.info("\n--- Step 3: Computing Train Gram Matrix & Fitting OC-SVM ---")
        t0 = time.time()
        K_train = parallel_gram_matrix(X_norm_train, n_jobs=args.n_jobs)
        logger.info(f"Train Gram matrix shape: {K_train.shape} computed in {time.time() - t0:.1f}s")
        
        svm = OneClassSVM(kernel="precomputed", nu=args.nu_param)
        svm.fit(K_train)
        del K_train
        gc.collect()
        logger.info(f"Training complete in {time.time() - t0:.1f}s")

        # ------------------------------------------------------------------
        # 4. Save the Model Immediately
        # ------------------------------------------------------------------
        logger.info("\n--- Step 4: Saving Trained Model ---")
        model_name = f"ocsvm_{fold_name}_exact.pkl"
        model_path = os.path.join(models_dir, model_name)
        artifact = ModelArtifact(
            model=svm,
            train_sequences=train_data,
            max_k=args.max_k,
            mismatches=args.mismatches,
            nu_param=args.nu_param,
            mkl_weights=mkl_weights,
            backend="precomputed",
            nystrom_state=None,
        )
        save_svm_model(artifact, model_path)
        logger.info(f"Saved model to {model_path}")
        
        del train_data, artifact
        gc.collect()

        # ------------------------------------------------------------------
        # 5. Testing Phase
        # ------------------------------------------------------------------
        logger.info("\n--- Step 5: Testing ---")
        for file_path, label in test_files:
            subject_name = "_".join(os.path.basename(file_path).split("_")[:2])
            logger.info(f"─── Testing: {subject_name} ({label.upper()}) ───")

            test_seqs, X_test_norm = test_features_cache[file_path]
            n_test = X_test_norm.shape[0]
            
            all_inverted_scores = []
            
            for start_idx in range(0, n_test, args.test_batch_size):
                end_idx = min(start_idx + args.test_batch_size, n_test)
                X_batch = X_test_norm[start_idx:end_idx]
                
                K_test_batch = parallel_asymmetric_gram_matrix(X_batch, X_norm_train, n_jobs=args.n_jobs)
                anomaly_scores = svm.decision_function(K_test_batch)
                inverted = -anomaly_scores  # higher = more anomalous
                all_inverted_scores.extend(inverted)
                
                del K_test_batch, anomaly_scores, inverted
            
            inverted = np.array(all_inverted_scores)

            # ------------------------------------------------------------------
            # 6. Save Scores
            # ------------------------------------------------------------------
            scores_path = os.path.join(out_dir, f"{subject_name}_scores_seed{args.seed}.csv")
            with open(scores_path, "w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["anomaly_score"])
                for score in inverted:
                    writer.writerow([score])
            logger.info(f"  Saved {len(inverted)} scores to {scores_path}")

            mean_score = float(np.mean(inverted))
            std_score = float(np.std(inverted))
            logger.info(f"  Mean anomaly score: {mean_score:.4f} ± {std_score:.4f}")

            summary_rows.append({
                "fold": fold_name,
                "subject": subject_name,
                "label": label,
                "mean_score": round(mean_score, 6),
                "std_score": round(std_score, 6),
                "n_sequences": len(test_seqs),
            })

            del all_inverted_scores
            gc.collect()
            
        del X_norm_train
        gc.collect()
            
    # ------------------------------------------------------------------
    # Save Summary
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 75)
    logger.info(" ALL FOLDS COMPLETED")
    logger.info(f" Total execution time: {time.time() - total_start:.1f}s")
    
    summary_path = os.path.join(base_out_dir, f"summary_seed{args.seed}.csv")
    file_mode = "a" if args.start_fold > 1 else "w"
    file_exists = os.path.isfile(summary_path)
    
    with open(summary_path, file_mode, newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["fold", "subject", "label", "mean_score", "std_score", "n_sequences"])
        if file_mode == "w" or not file_exists:
            writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    
    logger.info(f" Aggregate summary saved to {summary_path}")
    logger.info("=" * 75)


if __name__ == "__main__":
    main()
