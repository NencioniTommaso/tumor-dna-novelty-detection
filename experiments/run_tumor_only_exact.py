"""
run_tumor_only_exact.py
Loads a pre-trained exact kernel OC-SVM model and tests ONLY on tumor patients 
over multiple non-overlapping rounds.

Usage
-----
    python experiments/run_tumor_only_exact.py \
        --data-dir /home/paolo/conticello \
        --tumor-ids 1 2 3 4 5 6 7 8 10 11 12 13 14 15 16 17 18 19 20 \
        --model-name ocsvm_no_mismatch_nu02.pkl \
        --output-dir results/loo_multiround_nu02_exact/m_0/k_6/LOO_Healthy_7
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

from src.data_utils import sample_non_overlapping_rounds
from src.gram import compute_asymmetric_normalized_kernel, ensure_mkl_weights
from src.model_io import load_svm_model
from experiments.experiments_utils import (
    setup_logger,
    create_base_parser,
    add_data_dir_arg,
    add_cache_dir_arg,
    add_execution_args,
    validate_files_exist,
)

logger = setup_logger(__name__)

DEFAULT_N_ROUNDS = 7
DEFAULT_SEQS_PER_ROUND = 20_000

def _save_subject_scores_csv(round_scores: list[np.ndarray], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    n_rounds = len(round_scores)

    with open(output_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([f"round_{i + 1}" for i in range(n_rounds)])
        n_rows = max(len(s) for s in round_scores)
        for row_idx in range(n_rows):
            row = [
                str(round_scores[col][row_idx]) if row_idx < len(round_scores[col]) else ""
                for col in range(n_rounds)
            ]
            writer.writerow(row)
    logger.info(f"  Saved scores to {output_path}")

def _save_summary_csv(summary_rows: list[dict], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fieldnames = ["subject", "label", "round", "mean_score", "std_score", "n_sequences"]

    with open(output_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    logger.info(f"Summary saved to {output_path}")

def _test_subject_multi_round_exact(
    subject_name: str,
    label: str,
    rounds: list[list[str]],
    artifact,
    n_jobs: int,
) -> tuple[list[np.ndarray], list[dict]]:
    round_scores = []
    summary_rows = []
    mkl_weights = ensure_mkl_weights(artifact.max_k, artifact.mismatches, artifact.mkl_weights)

    for r_idx, round_seqs in enumerate(rounds):
        round_label = f"{subject_name} round {r_idx + 1}/{len(rounds)}"
        logger.info(f"  [{round_label}] Computing exact kernel for {len(round_seqs)} seqs ...")

        K_test = compute_asymmetric_normalized_kernel(
            test_seqs=round_seqs,
            train_states=artifact.train_states,
            max_k=artifact.max_k,
            mismatches=artifact.mismatches,
            mkl_weights=mkl_weights,
            n_jobs=n_jobs,
        )

        anomaly_scores = artifact.model.decision_function(K_test)
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
        del K_test, anomaly_scores
        gc.collect()

    return round_scores, summary_rows

def main():
    parser = create_base_parser("Test ONLY tumor patients (multi-round) using exact kernel")
    add_data_dir_arg(parser, required=True)
    add_cache_dir_arg(parser, project_root)
    add_execution_args(parser)
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
        help="Random seed for data loading (default: 42).",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for score CSVs (e.g., results/loo_multiround_nu02_exact/m_0/k_6/LOO_Healthy_7).",
    )
    parser.add_argument(
        "--model-name", type=str, default="ocsvm_no_mismatch_nu02.pkl",
        help="Name of the saved model file (default: ocsvm_no_mismatch_nu02.pkl).",
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

    # ── Load Model ────────────────────────────────────────────────────
    model_path = os.path.join(project_root, "models", args.model_name)
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        sys.exit(1)

    logger.info(f"--- Loading Exact Model from {model_path} ---")
    artifact = load_svm_model(model_path)
    
    if artifact.backend != "precomputed":
        logger.error(f"Expected backend 'precomputed', but got '{artifact.backend}'. Use Nystrom script for nystrom backend.")
        sys.exit(1)
        
    logger.info(f"Successfully loaded model (max_k={artifact.max_k}, mismatches={artifact.mismatches}, nu={artifact.nu_param})")

    # ── Build tumor file list ─────────────────────────────────────────
    tumor_files = []
    for tid in args.tumor_ids:
        fpath = os.path.join(args.data_dir, f"Colo_{tid}_merged_subset_1200000.fa")
        tumor_files.append((tid, fpath))

    missing = [f for _, f in tumor_files if not os.path.exists(f)]
    if missing:
        for f in missing:
            logger.warning(f"Tumor file not found, will skip: {f}")
        tumor_files = [(tid, f) for tid, f in tumor_files if os.path.exists(f)]

    if not tumor_files:
        logger.error("No valid tumor files to test. Exiting.")
        sys.exit(1)

    # ── Output directory ──────────────────────────────────────────────
    out_dir = args.output_dir
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

    # ── Banner ────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info(" TUMOR-ONLY MULTI-ROUND EXPERIMENT (EXACT KERNEL)")
    logger.info("=" * 65)
    logger.info(f"  Tumor patients   : {['Colo_' + str(tid) for tid, _ in tumor_files]}")
    logger.info(f"  Test rounds      : {args.n_rounds} × {args.seqs_per_round} seqs")
    logger.info(f"  Kernel           : max_k={artifact.max_k}, m={artifact.mismatches}")
    logger.info(f"  Output dir       : {out_dir}")
    logger.info("=" * 65)

    total_start = time.time()
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

        round_scores, summary_rows = _test_subject_multi_round_exact(
            subject_name=subject_name,
            label="tumor",
            rounds=rounds,
            artifact=artifact,
            n_jobs=args.n_jobs,
        )

        scores_path = os.path.join(out_dir, f"{subject_name}_scores_seed{args.seed}.csv")
        _save_subject_scores_csv(round_scores, scores_path)
        all_summary_rows.extend(summary_rows)

        del rounds, round_scores
        gc.collect()

    total_elapsed = time.time() - total_start

    logger.info("")
    logger.info("=" * 75)
    logger.info(" EXPERIMENT SUMMARY (TUMOR-ONLY EXACT)")
    logger.info("=" * 75)
    logger.info(f"{'Subject':<16s} {'Label':<16s} {'Round':<7s} {'Mean Score':>12s} {'Std':>10s}")
    logger.info("-" * 75)
    for row in all_summary_rows:
        logger.info(
            f"{row['subject']:<16s} {row['label']:<16s} {row['round']:<7d} "
            f"{row['mean_score']:>12.6f} {row['std_score']:>10.6f}"
        )
    logger.info("-" * 75)

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

    summary_path = os.path.join(out_dir, f"summary_tumor_only_seed{args.seed}.csv")
    
    # We append if the file exists to collate everything together, 
    # but we can also just append our new rows to the existing CSV if it exists.
    # The simplest is to just write a new file or append. Let's write a tumor-specific one.
    _save_summary_csv(all_summary_rows, summary_path)

if __name__ == "__main__":
    main()
