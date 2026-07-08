"""
run_tumor_only_loo_nystrom_cached.py
Loads pre-trained LOO Nyström OC-SVM models and tests ONLY on specified tumor
patients.  Features are extracted once, cached to disk as .npz files in the
experiment's cache directory, and reused across all 7 folds.

Usage
-----
    python experiments/run_tumor_only_loo_nystrom_cached.py \\
        --data-dir /path/to/fasta_data \\
        --results-dir results/full_loo_nystrom_cached/m_1/k_6 \\
        --tumor-ids 1 2 3 4 5 6 7 8 10 16 17 18 19 20
"""

import csv
import gc
import glob
import os
import sys
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import numpy as np
import scipy.sparse as sp

from src.data_utils import sample_non_overlapping_rounds
from src.gram import generate_mkl_weights
from src.mismatch import build_full_vocabulary
from src.model_io import load_svm_model
from src.nystrom import (
    build_combined_test_features,
    normalize_rows,
    nystrom_transform,
)
from experiments.experiments_utils import (
    setup_logger,
    create_base_parser,
    add_data_dir_arg,
    add_cache_dir_arg,
    add_execution_args,
)

logger = setup_logger(__name__)


def main():
    parser = create_base_parser(
        "Test tumor patients on pre-trained LOO Nyström models (disk-cached features)"
    )
    add_data_dir_arg(parser, required=True)
    add_cache_dir_arg(parser, project_root)
    add_execution_args(parser)
    parser.add_argument(
        "--tumor-ids", type=int, nargs="+", required=True,
        help="IDs of tumor patients to test (e.g. 1 2 3 4 5 6 7 8 10 16 17 18 19 20).",
    )
    parser.add_argument(
        "--results-dir", type=str, required=True,
        help="Path to the base results dir containing models/ and LOO_* dirs "
             "(e.g. results/full_loo_nystrom_cached/m_1/k_6).",
    )
    parser.add_argument(
        "--max-test", type=int, default=50_000,
        help="Sequences to infer per test subject (default: 50000).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for data loading (default: 42).",
    )
    parser.add_argument(
        "--skip-existing", action="store_true", default=True,
        help="Skip if score CSV already exists (default: True).",
    )
    parser.add_argument(
        "--no-skip-existing", dest="skip_existing", action="store_false",
        help="Force re-computation even if scores already exist.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Discover pre-trained models
    # ------------------------------------------------------------------
    models_dir = os.path.join(args.results_dir, "models")
    if not os.path.exists(models_dir):
        logger.error(f"Models directory not found: {models_dir}")
        sys.exit(1)

    model_files = sorted(
        glob.glob(os.path.join(models_dir, "ocsvm_LOO_Healthy_*_nystrom.pkl"))
    )
    if not model_files:
        logger.error(f"No pre-trained Nyström models found in {models_dir}")
        sys.exit(1)

    logger.info(f"Found {len(model_files)} pre-trained models in {models_dir}")

    # ------------------------------------------------------------------
    # 2. Load first model to extract kernel parameters
    # ------------------------------------------------------------------
    logger.info(f"Loading first model to extract kernel params: {os.path.basename(model_files[0])}")
    first_artifact = load_svm_model(model_files[0])
    max_k = first_artifact.max_k
    mismatches = first_artifact.mismatches
    mkl_weights = first_artifact.mkl_weights
    if mkl_weights is None:
        mkl_weights = generate_mkl_weights(max_k, noise_threshold=max(1, 2 * mismatches))
    logger.info(f"Kernel params: max_k={max_k}, mismatches={mismatches}, weights={mkl_weights}")
    del first_artifact
    gc.collect()

    # ------------------------------------------------------------------
    # 3. Build tumor file list
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 4. Locate experiment cache directory
    # ------------------------------------------------------------------
    cache_base = os.path.join(args.results_dir, "cache")
    if not os.path.isdir(cache_base):
        logger.error(f"Cache base directory not found: {cache_base}")
        sys.exit(1)

    # Find the existing nystrom_cache_* subdirectory
    cache_subdirs = [
        d for d in os.listdir(cache_base)
        if os.path.isdir(os.path.join(cache_base, d)) and d.startswith("nystrom_cache_")
    ]
    if len(cache_subdirs) == 1:
        experiment_cache_dir = os.path.join(cache_base, cache_subdirs[0])
    elif len(cache_subdirs) > 1:
        # Use the most recent one
        experiment_cache_dir = os.path.join(
            cache_base,
            sorted(cache_subdirs, key=lambda d: os.path.getmtime(os.path.join(cache_base, d)))[-1],
        )
        logger.warning(f"Multiple cache dirs found, using most recent: {experiment_cache_dir}")
    else:
        logger.error(f"No nystrom_cache_* subdirectory found in {cache_base}")
        sys.exit(1)

    logger.info(f"Using experiment cache directory: {experiment_cache_dir}")

    # ------------------------------------------------------------------
    # 5. Build vocabularies
    # ------------------------------------------------------------------
    logger.info("\n--- Building Vocabularies ---")
    start_k = max(1, mismatches + 1) if mismatches > 0 else 1
    active_ks = [k for k in range(start_k, max_k + 1) if mkl_weights[k - 1] != 0.0]
    per_k_vocabs = {}
    for k in active_ks:
        per_k_vocabs[k] = build_full_vocabulary(k)
        logger.info(f"Built full vocabulary for k={k} (size={len(per_k_vocabs[k])})")

    # ------------------------------------------------------------------
    # 6. Pre-cache tumor features to disk
    # ------------------------------------------------------------------
    logger.info("\n--- Pre-caching TUMOR features to disk ---")
    tumor_cache_info = {}  # tid -> (test_seqs, npz_path)
    t0_cache = time.time()

    for tid, file_path in tumor_files:
        subject_name = f"Colo_{tid}"
        npz_path = os.path.join(experiment_cache_dir, f"{subject_name}_test.npz")

        # Reuse from cache if already extracted (e.g. Colo_11 through Colo_15)
        if os.path.isfile(npz_path):
            logger.info(f"  Reusing cached features: {subject_name}")
            rounds = sample_non_overlapping_rounds(
                fasta_path=file_path,
                n_rounds=1,
                seqs_per_round=args.max_test,
                seed=args.seed,
                cache_dir=args.cache_dir,
                excluded_indices=None,
            )
            test_seqs = rounds[0]
            tumor_cache_info[tid] = (test_seqs, npz_path)
            continue

        logger.info(f"  Extracting test features for {subject_name} ...")
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

        sp.save_npz(npz_path, X_test_norm)
        logger.info(f"  Saved features to {npz_path} (shape={X_test_norm.shape})")
        del X_test_norm
        gc.collect()

        tumor_cache_info[tid] = (test_seqs, npz_path)

    logger.info(f"Tumor feature caching completed in {time.time() - t0_cache:.1f}s")

    # ------------------------------------------------------------------
    # Banner
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 65)
    logger.info(" TUMOR-ONLY TEST ON PRE-TRAINED LOO NYSTRÖM MODELS")
    logger.info("=" * 65)
    logger.info(f"  Models dir       : {models_dir}")
    logger.info(f"  Tumor patients   : {['Colo_' + str(tid) for tid, _ in tumor_files]}")
    logger.info(f"  Test sequences   : {args.max_test}")
    logger.info(f"  Kernel           : max_k={max_k}, m={mismatches}")
    logger.info(f"  MKL weights      : {mkl_weights}")
    logger.info(f"  Cache dir        : {experiment_cache_dir}")
    logger.info(f"  Skip existing    : {args.skip_existing}")
    logger.info("=" * 65)

    # ------------------------------------------------------------------
    # 7. Iterate over folds
    # ------------------------------------------------------------------
    summary_rows = []
    total_start = time.time()

    for model_path in model_files:
        model_name = os.path.basename(model_path)
        fold_name = model_name.replace("ocsvm_", "").replace("_nystrom.pkl", "")
        out_dir = os.path.join(args.results_dir, fold_name)
        os.makedirs(out_dir, exist_ok=True)

        # Check if all tumors already done for this fold
        if args.skip_existing:
            all_done = True
            for tid, _ in tumor_files:
                scores_path = os.path.join(out_dir, f"Colo_{tid}_scores_seed{args.seed}.csv")
                if not os.path.exists(scores_path):
                    all_done = False
                    break
            if all_done:
                logger.info(f"\nSkipping {fold_name} — all tumor scores already exist.")
                continue

        logger.info("\n" + "=" * 65)
        logger.info(f" FOLD: {fold_name}")
        logger.info("=" * 65)

        logger.info(f"Loading model: {model_name}")
        artifact = load_svm_model(model_path)
        svm = artifact.model
        nystrom_state = artifact.nystrom_state

        if nystrom_state is None:
            logger.error(f"Model {model_name} has no NystromState. Skipping fold.")
            del artifact, svm
            gc.collect()
            continue

        for tid, _ in tumor_files:
            subject_name = f"Colo_{tid}"
            scores_path = os.path.join(out_dir, f"{subject_name}_scores_seed{args.seed}.csv")

            if args.skip_existing and os.path.exists(scores_path):
                logger.info(f"  Skipping {subject_name} — already tested.")
                continue

            logger.info(f"─── Testing: {subject_name} ───")
            test_seqs, npz_path = tumor_cache_info[tid]

            # Load cached features from disk
            X_test_norm = sp.load_npz(npz_path)

            logger.info(f"  Projecting {X_test_norm.shape[0]} seqs into Nyström space ...")
            Phi_test = nystrom_transform(X_test_norm, nystrom_state, n_jobs=args.n_jobs)
            del X_test_norm

            anomaly_scores = svm.decision_function(Phi_test)
            inverted = -anomaly_scores  # higher = more anomalous
            del Phi_test, anomaly_scores

            # Save scores
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
                "label": "tumor",
                "mean_score": round(mean_score, 6),
                "std_score": round(std_score, 6),
                "n_sequences": len(test_seqs),
            })

            del inverted
            gc.collect()

        del artifact, svm, nystrom_state
        gc.collect()

    # ------------------------------------------------------------------
    # 8. Save aggregate summary
    # ------------------------------------------------------------------
    total_elapsed = time.time() - total_start

    logger.info("\n" + "=" * 75)
    logger.info(" ALL FOLDS COMPLETED")
    logger.info(f" Total execution time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")

    if summary_rows:
        logger.info("")
        logger.info(f"{'Fold':<18s} {'Subject':<16s} {'Label':<8s} {'Mean Score':>12s} {'Std':>10s}")
        logger.info("-" * 75)
        for row in summary_rows:
            logger.info(
                f"{row['fold']:<18s} {row['subject']:<16s} {row['label']:<8s} "
                f"{row['mean_score']:>12.6f} {row['std_score']:>10.6f}"
            )

        summary_path = os.path.join(args.results_dir, f"summary_tumor_only_seed{args.seed}.csv")
        file_mode = "a" if os.path.isfile(summary_path) else "w"
        file_exists = os.path.isfile(summary_path)

        with open(summary_path, file_mode, newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["fold", "subject", "label", "mean_score", "std_score", "n_sequences"],
            )
            if file_mode == "w" or not file_exists:
                writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)

        logger.info(f" Aggregate summary {'appended to' if file_mode == 'a' else 'saved to'} {summary_path}")
    else:
        logger.info(" No new scores computed (all already existed).")

    logger.info("=" * 75)


if __name__ == "__main__":
    main()
