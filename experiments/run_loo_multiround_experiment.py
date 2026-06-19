"""
run_loo_multiround_experiment.py
Leave-One-Out Multi-Round Non-Overlapping Testing Experiment.

Leaves out one healthy patient (default: Healthy_7), trains an OC-SVM on
the remaining 5 healthy patients using the exact mismatch string kernel,
then tests **7 rounds of 10k non-overlapping sequences** on each subject:

  - The held-out healthy patient (e.g. Healthy_7)
  - 3 tumor patients (Colo_11, Colo_12, Colo_13)
  - The 5 training healthy patients (with sequences disjoint from training)

Outputs one CSV per test subject (7 columns, one per round, 10k rows of
inverted anomaly scores) plus an aggregate summary CSV.
"""

import csv
import gc
import os
import sys
import time

# Dynamically resolve paths to ensure the script runs from anywhere
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
    mixed_string_kernel,
    normalize_gram,
    compute_asymmetric_normalized_kernel,
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
# Per-subject multi-round testing
# ─────────────────────────────────────────────────────────────────────

def _test_subject_multi_round(
    subject_name: str,
    label: str,
    rounds: list[list[str]],
    svm: OneClassSVM,
    train_states: dict,
    max_k: int,
    mismatches: int,
    mkl_weights: list[float],
    n_jobs: int,
) -> tuple[list[np.ndarray], list[dict]]:
    """Run inference on all rounds for a single test subject.

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
        logger.info(f"  [{round_label}] Computing asymmetric kernel ({len(round_seqs)} seqs) ...")

        K_test = compute_asymmetric_normalized_kernel(
            test_seqs=round_seqs,
            train_states=train_states,
            max_k=max_k,
            mismatches=mismatches,
            mkl_weights=mkl_weights,
            n_jobs=n_jobs,
        )

        anomaly_scores = svm.decision_function(K_test)
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

        # Free the test kernel immediately
        del K_test, anomaly_scores

    return round_scores, summary_rows


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = create_base_parser(
        "LOO Multi-Round Non-Overlapping Testing (Mismatch String Kernel)"
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
        help="ID of the healthy patient to hold out (2–7, default: 7).",
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
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for results (default: results/loo_multiround/m_<m>/k_<k>/<fold>/).",
    )
    parser.add_argument(
        "--model-name", type=str, default="ocsvm_loo_multiround.pkl",
        help="Name of the saved model file (default: ocsvm_loo_multiround.pkl).",
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
        project_root, "results", "loo_multiround",
        f"m_{args.mismatches}", f"k_{args.max_k}", fold_name,
    )
    os.makedirs(out_dir, exist_ok=True)

    # ── Banner ────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info(" LOO MULTI-ROUND NON-OVERLAPPING TESTING EXPERIMENT")
    logger.info("=" * 65)
    logger.info(f"  Fold             : {fold_name}")
    logger.info(f"  Training patients: {[os.path.basename(f) for f in fold['train_files']]}")
    logger.info(f"  Held-out healthy : {os.path.basename(fold['held_out_file'])}")
    logger.info(f"  Tumor patients   : {[os.path.basename(f) for f in fold['tumor_files']]}")
    logger.info(f"  Max train seqs   : {args.max_train}")
    logger.info(f"  Test rounds      : {args.n_rounds} × {args.seqs_per_round} seqs")
    logger.info(f"  Kernel           : max_k={args.max_k}, m={args.mismatches}")
    logger.info(f"  OCSVM nu         : {args.nu_param}")
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
    logger.info(f"Loaded {len(train_data)} training sequences from {len(fold['train_files'])} patients")

    # ══════════════════════════════════════════════════════════════════
    # STEP 2: Compute training kernel
    # ══════════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("--- Step 2: Computing Training Gram Matrix ---")
    mkl_weights = generate_mkl_weights(
        args.max_k, noise_threshold=max(1, 2 * args.mismatches)
    )
    logger.info(f"MKL weights: {mkl_weights}")

    K_train_unnorm, train_states = mixed_string_kernel(
        sequences=train_data,
        k_max=args.max_k,
        m=args.mismatches,
        weights=mkl_weights,
        n_jobs=args.n_jobs,
    )
    K_train = normalize_gram(K_train_unnorm)
    del K_train_unnorm
    gc.collect()

    # ══════════════════════════════════════════════════════════════════
    # STEP 3: Fit OC-SVM
    # ══════════════════════════════════════════════════════════════════
    logger.info("")
    logger.info(f"--- Step 3: Fitting One-Class SVM (nu={args.nu_param}) ---")
    svm = OneClassSVM(kernel="precomputed", nu=args.nu_param)
    svm.fit(K_train)
    del K_train
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
        train_states=train_states,
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

        round_scores, summary_rows = _test_subject_multi_round(
            subject_name=subject_name,
            label=label,
            rounds=rounds,
            svm=svm,
            train_states=train_states,
            max_k=args.max_k,
            mismatches=args.mismatches,
            mkl_weights=mkl_weights,
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

        round_scores, summary_rows = _test_subject_multi_round(
            subject_name=subject_name,
            label="healthy_train",
            rounds=rounds,
            svm=svm,
            train_states=train_states,
            max_k=args.max_k,
            mismatches=args.mismatches,
            mkl_weights=mkl_weights,
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
    logger.info(" EXPERIMENT SUMMARY")
    logger.info("=" * 75)
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
    logger.info(f"Total execution time: {total_elapsed:.1f}s")
    logger.info(f"Results written to: {out_dir}")

    # Save summary CSV
    summary_path = os.path.join(out_dir, f"summary_seed{args.seed}.csv")
    _save_summary_csv(all_summary_rows, summary_path)


if __name__ == "__main__":
    main()
