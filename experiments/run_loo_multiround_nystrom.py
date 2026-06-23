"""
run_loo_multiround_nystrom.py
Leave-One-Out Multi-Round Non-Overlapping Testing with Nyström Approximation.

Identical to run_loo_multiround_experiment.py, except it replaces the exact
N×N Gram matrix + precomputed-kernel OC-SVM with:

  1. Nyström approximation: selects m landmark points, computes an explicit
     feature map Φ ∈ ℝ^(N×m) such that Φ·Φᵀ ≈ K_norm
  2. Linear OC-SVM trained on Φ (instead of precomputed kernel)
  3. Test inference via Nyström projection (no N×N_test asymmetric kernel)

This reduces memory from O(N²) to O(N·m) and enables scaling to 100k+ sequences.

The m/N ratio defaults to 1% based on the m-selection experiment, which showed
that even m/N = 1% perfectly preserves AUC, rank correlation, and score gap.

Usage
-----
    python experiments/run_loo_multiround_nystrom.py \\
        --data-dir /path/to/fasta_files \\
        --max-k 6 --mismatches 1 --nu-param 0.2 --n-jobs -1

    # With custom m/N ratio:
    python experiments/run_loo_multiround_nystrom.py \\
        --data-dir /path/to/fasta_files \\
        --m-ratio 0.02 --max-k 6 --mismatches 1
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
DEFAULT_MAX_TRAIN = 30_000
DEFAULT_N_ROUNDS = 7
DEFAULT_SEQS_PER_ROUND = 10_000
DEFAULT_M_RATIO = 0.01
DEFAULT_LANDMARK_SEED = 42


# ─────────────────────────────────────────────────────────────────────
# Result I/O helpers
# ─────────────────────────────────────────────────────────────────────

def _save_subject_scores_csv(
    round_scores: list[np.ndarray],
    output_path: str,
) -> None:
    """Save per-round anomaly scores for one subject to a CSV.

    Creates a CSV with ``n_rounds`` columns (``round_1 … round_N``),
    where each column contains the inverted anomaly scores for one round.

    Parameters
    ----------
    round_scores : list[np.ndarray]
        One array per round; all arrays should have the same length.
    output_path : str
        Destination CSV path.
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

    Instead of computing an asymmetric kernel (N_test x N_train), this
    extracts features for the test round, normalizes, and projects into
    the Nyström feature space, then calls svm.decision_function on the
    resulting dense Φ_test matrix.

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
        "LOO Multi-Round Non-Overlapping Testing with Nyström Approximation"
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
        help="Output directory for results (default: results/loo_multiround_nystrom/m_<m>/k_<k>/<fold>/).",
    )
    parser.add_argument(
        "--model-name", type=str, default="ocsvm_loo_multiround_nystrom.pkl",
        help="Name of the saved model file (default: ocsvm_loo_multiround_nystrom.pkl).",
    )
    args = parser.parse_args()

    # ── Build fold ────────────────────────────────────────────────────
    fold = build_loo_single_fold(args.data_dir, args.held_out_id)
    fold_name = fold["fold_name"]

    # Validate all files exist
    all_files = fold["train_files"] + [fold["held_out_file"]] + fold["tumor_files"]
    if not validate_files_exist(all_files, logger):
        sys.exit(1)

    # ── Output directory ──────────────────────────────────────────────
    out_dir = args.output_dir or os.path.join(
        project_root, "results", "loo_multiround_nystrom",
        f"m_{args.mismatches}", f"k_{args.max_k}", fold_name,
    )
    os.makedirs(out_dir, exist_ok=True)

    # ── MKL weights ───────────────────────────────────────────────────
    mkl_weights = generate_mkl_weights(
        args.max_k, noise_threshold=max(1, 2 * args.mismatches)
    )

    # ── Banner ────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info(" LOO MULTI-ROUND EXPERIMENT (NYSTRÖM APPROXIMATION)")
    logger.info("=" * 65)
    logger.info(f"  Fold             : {fold_name}")
    logger.info(f"  Training patients: {[os.path.basename(f) for f in fold['train_files']]}")
    logger.info(f"  Held-out healthy : {os.path.basename(fold['held_out_file'])}")
    logger.info(f"  Tumor patients   : {[os.path.basename(f) for f in fold['tumor_files']]}")
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

    # Extract per-k sparse features directly (NO Gram matrix computation).
    # build_combined_feature_matrix only produces the sparse feature matrix
    # X_combined ∈ ℝ^(N × D_total), which is O(N·D) memory — not O(N²).
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

    # ══════════════════════════════════════════════════════════════════
    # STEP 5: Test non-training subjects (held-out healthy + tumors)
    # ══════════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=" * 65)
    logger.info(" TESTING NON-TRAINING SUBJECTS")
    logger.info("=" * 65)

    all_summary_rows: list[dict] = []

    # Subjects that were NOT in the training set — no excluded indices
    non_train_subjects = [
        (fold["held_out_file"], "healthy"),
    ] + [
        (f, "tumor") for f in fold["tumor_files"]
    ]

    for file_path, label in non_train_subjects:
        subject_name = "_".join(os.path.basename(file_path).split("_")[:2])
        logger.info("")
        logger.info(f"─── Testing: {subject_name} ({label.upper()}) ───")

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
            label=label,
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
    # STEP 6: Test training healthy patients (excluded training indices)
    # ══════════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=" * 65)
    logger.info(" TESTING TRAINING HEALTHY PATIENTS (NON-OVERLAPPING WITH TRAINING)")
    logger.info("=" * 65)

    for file_path in fold["train_files"]:
        subject_name = "_".join(os.path.basename(file_path).split("_")[:2])
        excluded = sampled_indices_per_file.get(file_path)
        logger.info("")
        logger.info(f"─── Testing: {subject_name} (HEALTHY — training patient) ───")
        if excluded is not None:
            logger.info(f"  Excluding {len(excluded)} training indices from sampling pool")

        rounds = sample_non_overlapping_rounds(
            fasta_path=file_path,
            n_rounds=args.n_rounds,
            seqs_per_round=args.seqs_per_round,
            seed=args.seed,
            cache_dir=args.cache_dir,
            excluded_indices=excluded,
        )

        round_scores, summary_rows = _test_subject_multi_round_nystrom(
            subject_name=subject_name,
            label="healthy_train",
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
    # STEP 7: Summary
    # ══════════════════════════════════════════════════════════════════
    total_elapsed = time.time() - total_start

    logger.info("")
    logger.info("=" * 75)
    logger.info(" EXPERIMENT SUMMARY (NYSTRÖM)")
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

    # Save summary CSV
    summary_path = os.path.join(out_dir, f"summary_seed{args.seed}.csv")
    _save_summary_csv(all_summary_rows, summary_path)


if __name__ == "__main__":
    main()
