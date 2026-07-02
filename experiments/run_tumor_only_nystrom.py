"""
run_tumor_only_nystrom.py
Trains an OC-SVM with Nyström approximation (identical to
run_loo_multiround_nystrom.py) and tests ONLY on tumor patients.

This is designed for the case where healthy patient scores already exist
from a prior experiment run, and you need to expand the tumor cohort
without re-testing healthy patients.

Training follows the same LOO protocol:
  - Hold out one healthy patient (default: Healthy_7)
  - Train on the remaining 6 healthy patients (max 120,000 sequences)
  - Nyström m/N = 0.01 (1%)

Testing:
  - Multi-round non-overlapping testing on each tumor patient
  - Saves per-subject score CSVs in the same format as the main experiment
  - Skips tumors whose score CSV already exists in the output directory

Usage
-----
    python experiments/run_tumor_only_nystrom.py \\
        --data-dir /home/paolo/conticello \\
        --tumor-ids 1 2 3 4 5 6 7 8 10 14 15 16 17 18 19 20 \\
        --output-dir results/loo_multiround_nystrom_001/m_1/k_6/LOO_Healthy_7
"""

import csv
import gc
import os
import sys
import time

# ── Path setup ─────────────────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import numpy as np
from sklearn.svm import OneClassSVM

from src.data_utils import (
    load_training_cohort_tracked_indices,
    sample_non_overlapping_rounds,
)
from src.gram import (
    generate_mkl_weights,
)
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
    build_loo_single_fold,
    validate_files_exist,
)

logger = setup_logger(__name__)

# ── Defaults ───────────────────────────────────────────────────────────
DEFAULT_MAX_TRAIN = 120_000
DEFAULT_N_ROUNDS = 7
DEFAULT_SEQS_PER_ROUND = 20_000
DEFAULT_M_RATIO = 0.01
DEFAULT_LANDMARK_SEED = 42


# ─────────────────────────────────────────────────────────────────────
# Result I/O helpers (identical to run_loo_multiround_nystrom.py)
# ─────────────────────────────────────────────────────────────────────

def _save_subject_scores_csv(
    round_scores: list[np.ndarray],
    output_path: str,
) -> None:
    """Save per-round anomaly scores for one subject to a CSV.

    Creates a CSV with ``n_rounds`` columns (``round_1 … round_N``),
    where each column contains the inverted anomaly scores for one round.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    n_rounds = len(round_scores)

    with open(output_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        # Header
        writer.writerow([f"round_{i + 1}" for i in range(n_rounds)])
        # Rows — all rounds are the same length (seqs_per_round)
        n_rows = max(len(s) for s in round_scores)
        for row_idx in range(n_rows):
            row = [
                str(round_scores[col][row_idx]) if row_idx < len(round_scores[col]) else ""
                for col in range(n_rounds)
            ]
            writer.writerow(row)

    logger.info(f"  Saved scores to {output_path}")


def _save_summary_csv(
    summary_rows: list[dict],
    output_path: str,
) -> None:
    """Save the aggregate summary CSV."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fieldnames = ["subject", "label", "round", "mean_score", "std_score", "n_sequences"]

    with open(output_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    logger.info(f"Summary saved to {output_path}")


# ─────────────────────────────────────────────────────────────────────
# Per-subject multi-round testing (Nyström version)
# ─────────────────────────────────────────────────────────────────────

def _test_subject_multi_round_nystrom(
    subject_name: str,
    label: str,
    rounds: list[list[str]],
    svm: OneClassSVM,
    nystrom_state: NystromState,
    n_jobs: int,
) -> tuple[list[np.ndarray], list[dict]]:
    """Run inference on all rounds for a single test subject using Nyström.

    Returns
    -------
    round_scores : list[np.ndarray]
        Inverted anomaly scores per round (higher = more anomalous).
    summary_rows : list[dict]
        One summary dict per round.
    """
    round_scores: list[np.ndarray] = []
    summary_rows: list[dict] = []

    for r_idx, round_seqs in enumerate(rounds):
        round_label = f"{subject_name} round {r_idx + 1}/{len(rounds)}"
        logger.info(f"  [{round_label}] Projecting {len(round_seqs)} seqs into Nyström space ...")

        # Build combined test features using training vocabularies
        X_test_combined = build_combined_test_features(
            round_seqs, nystrom_state.per_k_vocabs,
            nystrom_state.max_k, nystrom_state.mismatches,
            nystrom_state.mkl_weights,
            n_jobs=n_jobs,
        )
        X_test_norm, _ = normalize_rows(X_test_combined)
        del X_test_combined

        # Project into Nyström feature space
        Phi_test = nystrom_transform(X_test_norm, nystrom_state, n_jobs=n_jobs)
        del X_test_norm

        # Score with linear OC-SVM
        anomaly_scores = svm.decision_function(Phi_test)
        inverted = -anomaly_scores  # higher = more anomalous
        round_scores.append(inverted)

        mean_score = float(np.mean(inverted))
        std_score = float(np.std(inverted))

        summary_rows.append({
            "subject": subject_name,
            "label": label,
            "round": r_idx + 1,
            "mean_score": round(mean_score, 6),
            "std_score": round(std_score, 6),
            "n_sequences": len(round_seqs),
        })

        logger.info(f"  [{round_label}] Mean anomaly score: {mean_score:.4f} ± {std_score:.4f}")

        del Phi_test, anomaly_scores

    return round_scores, summary_rows


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = create_base_parser(
        "Train Nyström OC-SVM and test ONLY tumor patients (multi-round)"
    )
    add_data_dir_arg(parser, required=True)
    add_cache_dir_arg(parser, project_root)
    add_kernel_args(parser)
    add_nu_arg(parser)
    add_execution_args(parser)
    parser.add_argument(
        "--max-train", type=int, default=DEFAULT_MAX_TRAIN,
        help=f"Total training sequences across all training patients (default: {DEFAULT_MAX_TRAIN}).",
    )
    parser.add_argument(
        "--held-out-id", type=int, default=7,
        help="ID of the healthy patient to hold out (1–7, default: 7).",
    )
    parser.add_argument(
        "--n-rounds", type=int, default=DEFAULT_N_ROUNDS,
        help=f"Number of non-overlapping test rounds (default: {DEFAULT_N_ROUNDS}).",
    )
    parser.add_argument(
        "--seqs-per-round", type=int, default=DEFAULT_SEQS_PER_ROUND,
        help=f"Sequences per test round (default: {DEFAULT_SEQS_PER_ROUND}).",
    )
    parser.add_argument(
        "--m-ratio", type=float, default=DEFAULT_M_RATIO,
        help=f"Nyström landmark ratio m/N (default: {DEFAULT_M_RATIO}).",
    )
    parser.add_argument(
        "--landmark-seed", type=int, default=DEFAULT_LANDMARK_SEED,
        help=f"Random seed for Nyström landmark selection (default: {DEFAULT_LANDMARK_SEED}).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for data loading (default: 42).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for score CSVs (default: results/loo_multiround_nystrom_001/m_<m>/k_<k>/LOO_Healthy_<id>/).",
    )
    parser.add_argument(
        "--model-name", type=str, default="ocsvm_tumor_only_nystrom.pkl",
        help="Name of the saved model file (default: ocsvm_tumor_only_nystrom.pkl).",
    )
    parser.add_argument(
        "--tumor-ids", type=int, nargs="+", required=True,
        help="IDs of tumor patients to test (e.g. 1 2 3 4 5 6 7 8 10 14 15 16 17 18 19 20).",
    )
    parser.add_argument(
        "--skip-existing", action="store_true", default=True,
        help="Skip tumor patients whose score CSV already exists in the output directory (default: True).",
    )
    parser.add_argument(
        "--no-skip-existing", dest="skip_existing", action="store_false",
        help="Force re-computation of all tumor patients.",
    )
    args = parser.parse_args()

    # ── Build fold (for training data only) ───────────────────────────
    fold = build_loo_single_fold(args.data_dir, args.held_out_id)
    fold_name = fold["fold_name"]

    # Validate training files exist
    if not validate_files_exist(fold["train_files"], logger):
        sys.exit(1)

    # ── Build tumor file list ─────────────────────────────────────────
    tumor_files = []
    for tid in args.tumor_ids:
        fpath = os.path.join(args.data_dir, f"Colo_{tid}_merged_subset_1200000.fa")
        tumor_files.append((tid, fpath))

    # Validate tumor files exist
    missing = [f for _, f in tumor_files if not os.path.exists(f)]
    if missing:
        for f in missing:
            logger.warning(f"Tumor file not found, will skip: {f}")
        tumor_files = [(tid, f) for tid, f in tumor_files if os.path.exists(f)]

    if not tumor_files:
        logger.error("No valid tumor files to test. Exiting.")
        sys.exit(1)

    # ── Output directory ──────────────────────────────────────────────
    out_dir = args.output_dir or os.path.join(
        project_root, "results", "loo_multiround_nystrom_001",
        f"m_{args.mismatches}", f"k_{args.max_k}", fold_name,
    )
    os.makedirs(out_dir, exist_ok=True)

    # ── Check which tumors to skip ────────────────────────────────────
    if args.skip_existing:
        to_test = []
        for tid, fpath in tumor_files:
            scores_csv = os.path.join(out_dir, f"Colo_{tid}_scores_seed{args.seed}.csv")
            if os.path.exists(scores_csv):
                logger.info(f"Skipping Colo_{tid} — scores already exist: {scores_csv}")
            else:
                to_test.append((tid, fpath))
        tumor_files = to_test

    if not tumor_files:
        logger.info("All requested tumor patients already have scores. Nothing to do.")
        sys.exit(0)

    # ── MKL weights ───────────────────────────────────────────────────
    mkl_weights = generate_mkl_weights(
        args.max_k, noise_threshold=max(1, 2 * args.mismatches)
    )

    # ── Banner ────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info(" TUMOR-ONLY MULTI-ROUND EXPERIMENT (NYSTRÖM APPROXIMATION)")
    logger.info("=" * 65)
    logger.info(f"  Fold             : {fold_name}")
    logger.info(f"  Training patients: {[os.path.basename(f) for f in fold['train_files']]}")
    logger.info(f"  Tumor patients   : {['Colo_' + str(tid) for tid, _ in tumor_files]}")
    logger.info(f"  Max train seqs   : {args.max_train}")
    logger.info(f"  Test rounds      : {args.n_rounds} × {args.seqs_per_round} seqs")
    logger.info(f"  Kernel           : max_k={args.max_k}, m={args.mismatches}")
    logger.info(f"  MKL weights      : {mkl_weights}")
    logger.info(f"  OCSVM nu         : {args.nu_param}")
    logger.info(f"  Nyström m/N      : {args.m_ratio:.2%}")
    logger.info(f"  Landmark seed    : {args.landmark_seed}")
    logger.info(f"  Seed             : {args.seed}")
    logger.info(f"  Output dir       : {out_dir}")
    logger.info("=" * 65)

    total_start = time.time()

    # ══════════════════════════════════════════════════════════════════
    # STEP 1: Load training data with index tracking
    # ══════════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("--- Step 1: Loading Training Data (with index tracking) ---")
    train_data, sampled_indices_per_file = load_training_cohort_tracked_indices(
        fold["train_files"], args.max_train, args.seed, args.cache_dir,
    )
    N = len(train_data)
    logger.info(f"Loaded {N:,} training sequences from {len(fold['train_files'])} patients")

    # ══════════════════════════════════════════════════════════════════
    # STEP 2: Extract features and build Nyström approximation
    # ══════════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("--- Step 2: Extracting Features & Building Nyström Approximation ---")

    t0 = time.time()

    X_combined, per_k_vocabs = build_combined_feature_matrix(
        sequences=train_data,
        k_max=args.max_k,
        mismatches=args.mismatches,
        mkl_weights=mkl_weights,
        n_jobs=args.n_jobs,
    )

    # Row-normalize (cosine normalization)
    X_norm_train, _ = normalize_rows(X_combined)
    del X_combined
    gc.collect()
    logger.info("Training features row-normalized")

    # Compute m from ratio
    m = max(1, int(args.m_ratio * N))
    logger.info(f"Nyström: m = {m:,} landmarks (m/N = {args.m_ratio:.2%})")

    # Fit Nyström
    nystrom_state = nystrom_fit(
        X_norm_train, m, args.landmark_seed,
        per_k_vocabs, mkl_weights,
        args.max_k, args.mismatches,
    )

    # Transform training data into Nyström feature space
    Phi_train = nystrom_transform(X_norm_train, nystrom_state, n_jobs=args.n_jobs)
    del X_norm_train
    gc.collect()

    feature_time = time.time() - t0
    logger.info(f"Feature extraction + Nyström fit/transform complete in {feature_time:.1f}s")
    logger.info(f"Φ_train shape: {Phi_train.shape[0]:,} × {Phi_train.shape[1]:,}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 3: Fit linear OC-SVM on Nyström features
    # ══════════════════════════════════════════════════════════════════
    logger.info("")
    logger.info(f"--- Step 3: Fitting Linear OC-SVM on Nyström Features (nu={args.nu_param}) ---")
    svm = OneClassSVM(kernel="linear", nu=args.nu_param)
    svm.fit(Phi_train)
    del Phi_train
    gc.collect()

    train_elapsed = time.time() - total_start
    logger.info(f"Training complete in {train_elapsed:.1f}s")

    # ══════════════════════════════════════════════════════════════════
    # STEP 4: Save the trained model
    # ══════════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("--- Step 4: Saving Trained Model ---")
    model_dir = os.path.join(project_root, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, args.model_name)

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

    # Free training data
    del train_data
    gc.collect()

    # ══════════════════════════════════════════════════════════════════
    # STEP 5: Test tumor patients only
    # ══════════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=" * 65)
    logger.info(" TESTING TUMOR PATIENTS")
    logger.info("=" * 65)

    all_summary_rows: list[dict] = []

    for tid, file_path in tumor_files:
        subject_name = f"Colo_{tid}"
        logger.info("")
        logger.info(f"─── Testing: {subject_name} (TUMOR) ───")

        rounds = sample_non_overlapping_rounds(
            fasta_path=file_path,
            n_rounds=args.n_rounds,
            seqs_per_round=args.seqs_per_round,
            seed=args.seed,
            cache_dir=args.cache_dir,
            excluded_indices=None,
        )

        round_scores, summary_rows = _test_subject_multi_round_nystrom(
            subject_name=subject_name,
            label="tumor",
            rounds=rounds,
            svm=svm,
            nystrom_state=nystrom_state,
            n_jobs=args.n_jobs,
        )

        scores_path = os.path.join(out_dir, f"{subject_name}_scores_seed{args.seed}.csv")
        _save_subject_scores_csv(round_scores, scores_path)
        all_summary_rows.extend(summary_rows)

        del rounds, round_scores
        gc.collect()

    # ══════════════════════════════════════════════════════════════════
    # STEP 6: Summary
    # ══════════════════════════════════════════════════════════════════
    total_elapsed = time.time() - total_start

    logger.info("")
    logger.info("=" * 75)
    logger.info(" EXPERIMENT SUMMARY (TUMOR-ONLY NYSTRÖM)")
    logger.info("=" * 75)
    logger.info(f"  Nyström m/N = {args.m_ratio:.2%}  (m = {m:,}, N = {N:,})")
    logger.info(f"  Landmark seed = {args.landmark_seed}")
    logger.info("")
    logger.info(f"{'Subject':<16s} {'Label':<16s} {'Round':<7s} {'Mean Score':>12s} {'Std':>10s}")
    logger.info("-" * 75)
    for row in all_summary_rows:
        logger.info(
            f"{row['subject']:<16s} {row['label']:<16s} {row['round']:<7d} "
            f"{row['mean_score']:>12.6f} {row['std_score']:>10.6f}"
        )
    logger.info("-" * 75)

    # Per-subject aggregate across rounds
    subjects_seen = []
    for row in all_summary_rows:
        if row["subject"] not in [s[0] for s in subjects_seen]:
            subjects_seen.append((row["subject"], row["label"]))

    logger.info("")
    logger.info(f"{'Subject':<16s} {'Label':<16s} {'Mean (across rounds)':>22s} {'Std (across rounds)':>22s}")
    logger.info("-" * 75)
    for subj, label in subjects_seen:
        subj_means = [r["mean_score"] for r in all_summary_rows if r["subject"] == subj]
        logger.info(
            f"{subj:<16s} {label:<16s} "
            f"{np.mean(subj_means):>22.6f} {np.std(subj_means):>22.6f}"
        )

    logger.info("=" * 75)
    logger.info(f"Total execution time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    logger.info(f"Results written to: {out_dir}")

    # Save tumor-only summary CSV
    summary_path = os.path.join(out_dir, f"summary_tumor_only_seed{args.seed}.csv")
    _save_summary_csv(all_summary_rows, summary_path)


if __name__ == "__main__":
    main()
