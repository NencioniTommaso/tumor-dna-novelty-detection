"""
run_full_loo_nystrom.py
Leave-One-Out Cross Validation for Nyström Approximation over all 7 healthy patients.
For each fold, one healthy patient is held out. Models are evaluated on the 
held-out healthy patient and a set of 5 specific tumor files.
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
from sklearn.svm import OneClassSVM

from src.data_utils import (
    load_train_cohort_only,
    sample_non_overlapping_rounds,
)
from src.gram import generate_mkl_weights
from src.nystrom import (
    NystromState,
    build_combined_feature_matrix,
    build_combined_test_features,
    normalize_rows,
    nystrom_fit,
    nystrom_transform,
)
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
    parser = create_base_parser("Full LOO Experiment with Nyström")
    add_data_dir_arg(parser, required=True)
    add_cache_dir_arg(parser, project_root)
    add_kernel_args(parser)
    add_nu_arg(parser)
    add_execution_args(parser)
    parser.add_argument("--max-train", type=int, default=120_000,
                        help="Total training sequences across all training patients (default: 120000).")
    parser.add_argument("--max-test", type=int, default=50_000,
                        help="Sequences to infer per test subject (default: 50000).")
    parser.add_argument("--m-ratio", type=float, default=0.01,
                        help="Nyström landmark ratio m/N (default: 0.01).")
    parser.add_argument("--landmark-seed", type=int, default=42,
                        help="Random seed for Nyström landmark selection (default: 42).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for data loading (default: 42).")
    args = parser.parse_args()

    all_healthy = build_all_healthy_files(args.data_dir)
    colo_files = [os.path.join(args.data_dir, f"Colo_{i}_merged_subset_1200000.fa") for i in range(COLO_START, COLO_END + 1)]
    
    # Validation
    if not validate_files_exist(all_healthy + colo_files, logger):
        sys.exit(1)

    mkl_weights = generate_mkl_weights(args.max_k, noise_threshold=max(1, 2 * args.mismatches))

    # Output directories
    base_out_dir = os.path.join(
        project_root, "results", "full_loo_nystrom", 
        f"m_{args.mismatches}", f"k_{args.max_k}"
    )
    models_dir = os.path.join(base_out_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    logger.info("=" * 65)
    logger.info(" FULL LOO EXPERIMENT (NYSTRÖM APPROXIMATION)")
    logger.info("=" * 65)
    logger.info(f"  Colo Range       : {COLO_START} to {COLO_END}")
    logger.info(f"  Max train seqs   : {args.max_train}")
    logger.info(f"  Test sequences   : {args.max_test}")
    logger.info(f"  Kernel           : max_k={args.max_k}, m={args.mismatches}")
    logger.info(f"  MKL weights      : {mkl_weights}")
    logger.info(f"  OCSVM nu         : {args.nu_param}")
    logger.info(f"  Nyström m/N      : {args.m_ratio:.2%}")
    logger.info(f"  Output dir       : {base_out_dir}")
    logger.info("=" * 65)

    summary_rows = []
    total_start = time.time()

    for held_out_id in range(1, 8):
        fold_name = f"LOO_Healthy_{held_out_id}"
        out_dir = os.path.join(base_out_dir, fold_name)
        os.makedirs(out_dir, exist_ok=True)

        logger.info("")
        logger.info("=" * 65)
        logger.info(f" STARTING FOLD: {fold_name}")
        logger.info("=" * 65)

        # Separate train / test files
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
        # 1. Load Training Data
        # ------------------------------------------------------------------
        logger.info("\n--- Step 1: Loading Training Data ---")
        train_data = load_train_cohort_only(train_files, args.max_train, args.seed, args.cache_dir)
        N = len(train_data)
        logger.info(f"Loaded {N:,} training sequences.")

        # ------------------------------------------------------------------
        # 2. Extract Features & Build Nyström Approximation
        # ------------------------------------------------------------------
        logger.info("\n--- Step 2: Extracting Features & Building Nyström Approximation ---")
        t0 = time.time()
        X_combined, per_k_vocabs = build_combined_feature_matrix(
            sequences=train_data,
            k_max=args.max_k,
            mismatches=args.mismatches,
            mkl_weights=mkl_weights,
            n_jobs=args.n_jobs,
        )
        X_norm_train, _ = normalize_rows(X_combined)
        del X_combined
        gc.collect()

        m = max(1, int(args.m_ratio * N))
        logger.info(f"Nyström: m = {m:,} landmarks (m/N = {args.m_ratio:.2%})")

        nystrom_state = nystrom_fit(
            X_norm_train, m, args.landmark_seed,
            per_k_vocabs, mkl_weights,
            args.max_k, args.mismatches,
        )

        Phi_train = nystrom_transform(X_norm_train, nystrom_state, n_jobs=args.n_jobs)
        del X_norm_train
        gc.collect()
        logger.info(f"Feature extraction + Nyström fit/transform complete in {time.time() - t0:.1f}s")

        # ------------------------------------------------------------------
        # 3. Fit Linear OC-SVM
        # ------------------------------------------------------------------
        logger.info("\n--- Step 3: Fitting Linear OC-SVM ---")
        t0 = time.time()
        svm = OneClassSVM(kernel="linear", nu=args.nu_param)
        svm.fit(Phi_train)
        del Phi_train
        gc.collect()
        logger.info(f"Training complete in {time.time() - t0:.1f}s")

        # ------------------------------------------------------------------
        # 4. Save the Model Immediately
        # ------------------------------------------------------------------
        logger.info("\n--- Step 4: Saving Trained Model ---")
        model_name = f"ocsvm_{fold_name}_nystrom.pkl"
        model_path = os.path.join(models_dir, model_name)
        artifact = ModelArtifact(
            model=svm,
            train_sequences=train_data,
            max_k=args.max_k,
            mismatches=args.mismatches,
            nu_param=args.nu_param,
            mkl_weights=mkl_weights,
            backend="nystrom",
            nystrom_state=nystrom_state,
        )
        save_svm_model(artifact, model_path)
        logger.info(f"Saved model to {model_path}")
        
        # We don't need train_data or artifact anymore in memory
        del train_data, artifact
        gc.collect()

        # ------------------------------------------------------------------
        # 5. Testing Phase (Test 50k sequences for each subject)
        # ------------------------------------------------------------------
        logger.info("\n--- Step 5: Testing ---")
        for file_path, label in test_files:
            subject_name = "_".join(os.path.basename(file_path).split("_")[:2])
            logger.info(f"─── Testing: {subject_name} ({label.upper()}) ───")

            rounds = sample_non_overlapping_rounds(
                fasta_path=file_path,
                n_rounds=1,
                seqs_per_round=args.max_test,
                seed=args.seed,
                cache_dir=args.cache_dir,
                excluded_indices=None,
            )
            test_seqs = rounds[0]

            logger.info(f"  Projecting {len(test_seqs)} seqs into Nyström space ...")
            X_test_combined = build_combined_test_features(
                test_seqs, nystrom_state.per_k_vocabs,
                nystrom_state.max_k, nystrom_state.mismatches,
                nystrom_state.mkl_weights,
                n_jobs=args.n_jobs,
            )
            X_test_norm, _ = normalize_rows(X_test_combined)
            del X_test_combined

            Phi_test = nystrom_transform(X_test_norm, nystrom_state, n_jobs=args.n_jobs)
            del X_test_norm

            anomaly_scores = svm.decision_function(Phi_test)
            inverted = -anomaly_scores  # higher = more anomalous

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

            del Phi_test, anomaly_scores, inverted, rounds, test_seqs
            gc.collect()
            
    # ------------------------------------------------------------------
    # Save Summary
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 75)
    logger.info(" ALL FOLDS COMPLETED")
    logger.info(f" Total execution time: {time.time() - total_start:.1f}s")
    
    summary_path = os.path.join(base_out_dir, f"summary_seed{args.seed}.csv")
    with open(summary_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["fold", "subject", "label", "mean_score", "std_score", "n_sequences"])
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    
    logger.info(f" Aggregate summary saved to {summary_path}")
    logger.info("=" * 75)


if __name__ == "__main__":
    main()
