"""
run_tumor_only_loo_exact_cached.py
Loads pre-trained exact kernel OC-SVM models from a LOO experiment and tests 
ONLY on specified tumor patients. It caches tumor features to avoid redundant extraction.

Usage
-----
    python experiments/run_tumor_only_loo_exact_cached.py \
        --data-dir /path/to/data \
        --results-dir results/full_loo_exact_cached/m_0/k_6 \
        --tumor-ids 1 2 3 4 5 6 7 8 10 16 17 18 19 20
"""

import csv
import gc
import os
import sys
import time
import glob

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import numpy as np

from src.data_utils import sample_non_overlapping_rounds
from src.gram import parallel_asymmetric_gram_matrix, ensure_mkl_weights
from src.nystrom import build_combined_test_features, normalize_rows
from src.mismatch import build_full_vocabulary
from src.model_io import load_svm_model

from experiments.experiments_utils import (
    setup_logger,
    create_base_parser,
    add_data_dir_arg,
    add_cache_dir_arg,
    add_execution_args,
)

logger = setup_logger(__name__)

def main():
    parser = create_base_parser("Test tumor patients on pre-trained LOO exact models (Cached)")
    add_data_dir_arg(parser, required=True)
    add_cache_dir_arg(parser, project_root)
    add_execution_args(parser)
    parser.add_argument("--tumor-ids", type=int, nargs="+", required=True,
                        help="IDs of tumor patients to test (e.g. 1 2 3 4 5 6 7 8 10 16 17 18 19 20).")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Path to the base results dir containing models/ and LOO_* dirs.")
    parser.add_argument("--max-test", type=int, default=50_000,
                        help="Sequences to infer per test subject (default: 50000).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for data loading (default: 42).")
    parser.add_argument("--test-batch-size", type=int, default=10_000,
                        help="Number of test sequences to process at once (default: 10000).")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip if score CSV already exists (default: True).")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = parser.parse_args()

    models_dir = os.path.join(args.results_dir, "models")
    if not os.path.exists(models_dir):
        logger.error(f"Models directory not found: {models_dir}")
        sys.exit(1)

    model_files = sorted(glob.glob(os.path.join(models_dir, "ocsvm_LOO_Healthy_*_exact.pkl")))
    if not model_files:
        logger.error(f"No pre-trained exact models found in {models_dir}")
        sys.exit(1)

    # 1. Load the first model to get kernel params
    logger.info(f"Loading first model to extract kernel parameters: {os.path.basename(model_files[0])}")
    first_artifact = load_svm_model(model_files[0])
    max_k = first_artifact.max_k
    mismatches = first_artifact.mismatches
    mkl_weights = ensure_mkl_weights(max_k, mismatches, first_artifact.mkl_weights)
    logger.info(f"Kernel params: max_k={max_k}, mismatches={mismatches}, weights={mkl_weights}")
    del first_artifact
    gc.collect()

    # 2. Build tumor file list
    tumor_files = []
    for tid in args.tumor_ids:
        fpath = os.path.join(args.data_dir, f"Colo_{tid}_merged_subset_1200000.fa")
        if os.path.exists(fpath):
            tumor_files.append((tid, fpath))
        else:
            logger.warning(f"Tumor file not found, skipping: {fpath}")

    if not tumor_files:
        logger.error("No valid tumor files to test. Exiting.")
        sys.exit(1)

    # 3. Pre-cache vocabularies and tumor features
    logger.info("\n--- Building Vocabularies ---")
    start_k = max(1, mismatches + 1) if mismatches > 0 else 1
    active_ks = [k for k in range(start_k, max_k + 1) if mkl_weights[k - 1] != 0.0]
    per_k_vocabs = {}
    for k in active_ks:
        per_k_vocabs[k] = build_full_vocabulary(k)

    logger.info("\n--- Pre-caching TUMOR features ---")
    tumor_features_cache = {}
    t0_cache = time.time()
    
    for tid, file_path in tumor_files:
        subject_name = f"Colo_{tid}"
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
            max_k, mismatches, mkl_weights,
            n_jobs=args.n_jobs,
        )
        X_test_norm, _ = normalize_rows(X_test_combined)
        del X_test_combined
        
        tumor_features_cache[tid] = (test_seqs, X_test_norm)
        
    logger.info(f"Tumor feature caching completed in {time.time() - t0_cache:.1f}s")

    summary_rows = []
    total_start = time.time()

    # 4. Iterate over folds
    for model_path in model_files:
        model_name = os.path.basename(model_path)
        fold_name = model_name.replace("ocsvm_", "").replace("_exact.pkl", "")
        out_dir = os.path.join(args.results_dir, fold_name)
        os.makedirs(out_dir, exist_ok=True)

        logger.info("\n" + "=" * 65)
        logger.info(f" FOLD: {fold_name}")
        logger.info("=" * 65)

        logger.info(f"Loading model: {model_name}")
        artifact = load_svm_model(model_path)
        svm = artifact.model
        train_seqs = artifact.train_sequences

        logger.info(f"Extracting features for {len(train_seqs)} training sequences...")
        t0 = time.time()
        X_train_combined = build_combined_test_features(
            train_seqs, per_k_vocabs,
            max_k, mismatches, mkl_weights,
            n_jobs=args.n_jobs,
        )
        X_norm_train, _ = normalize_rows(X_train_combined)
        del X_train_combined
        logger.info(f"Training features extracted in {time.time() - t0:.1f}s")

        for tid, _ in tumor_files:
            subject_name = f"Colo_{tid}"
            scores_path = os.path.join(out_dir, f"{subject_name}_scores_seed{args.seed}.csv")

            if args.skip_existing and os.path.exists(scores_path):
                logger.info(f"  Skipping {subject_name} — already tested.")
                continue

            logger.info(f"─── Testing: {subject_name} ───")
            test_seqs, X_test_norm = tumor_features_cache[tid]
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

            with open(scores_path, "w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["anomaly_score"])
                for score in inverted:
                    writer.writerow([score])
            logger.info(f"  Saved {len(inverted)} scores to {scores_path}")

            mean_score = float(np.mean(inverted))
            std_score = float(np.std(inverted))
            logger.info(f"  Mean score: {mean_score:.4f} ± {std_score:.4f}")

            summary_rows.append({
                "fold": fold_name,
                "subject": subject_name,
                "label": "tumor",
                "mean_score": round(mean_score, 6),
                "std_score": round(std_score, 6),
                "n_sequences": len(test_seqs),
            })

            del all_inverted_scores

        del X_norm_train, artifact, svm, train_seqs
        gc.collect()

    logger.info("\n" + "=" * 75)
    logger.info(" ALL FOLDS COMPLETED")
    logger.info(f" Total execution time: {time.time() - total_start:.1f}s")
    
    if summary_rows:
        summary_path = os.path.join(args.results_dir, f"summary_tumor_only_seed{args.seed}.csv")
        file_mode = "a" if os.path.isfile(summary_path) else "w"
        with open(summary_path, file_mode, newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["fold", "subject", "label", "mean_score", "std_score", "n_sequences"])
            if file_mode == "w":
                writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)
        logger.info(f" Aggregate summary appended to {summary_path}")
    logger.info("=" * 75)

if __name__ == "__main__":
    main()
