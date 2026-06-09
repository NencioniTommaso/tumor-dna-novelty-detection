"""
run_loo_experiment.py
Leave-One-Out (LOO) Cross-Validation over healthy patients for the
String-Kernel MIL pipeline.

For each of the 6 healthy patients (Healthy_2 … Healthy_7):
  - Train on the remaining 5 healthy patients.
  - Test the held-out healthy patient together with one tumor patient
    (passed via --tumor-file).

Produces per-fold patient-level AUC, per-patient score-distribution
plots, and an aggregate summary (mean ± std) written to CSV.
"""

import time
import sys
import os

# Dynamically resolve paths to ensure the script runs from anywhere
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import numpy as np

# Import from our custom library
from src.data_utils import load_tracked_patient_cohort
from src.gram import (
    generate_mkl_weights,
    mixed_string_kernel,
    normalize_gram,
    compute_asymmetric_normalized_kernel,
)
from src.evaluation import evaluate_novelty_detector, evaluate_patient_level_novelty
from src.loo_results import save_loo_results, print_loo_summary
from experiments.experiments_utils import (
    setup_logger,
    create_base_parser,
    add_data_dir_arg,
    add_cache_dir_arg,
    add_train_sampling_arg,
    add_seed_arg,
    add_kernel_args,
    add_nu_arg,
    add_execution_args,
    add_loo_args,
    build_loo_folds,
    validate_files_exist,
)

logger = setup_logger(__name__)


# ------------------------------------------------------------------
# Single-fold runner
# ------------------------------------------------------------------

def _run_single_fold(fold: dict, args) -> dict:
    """Execute one LOO fold and return its metrics.

    Parameters
    ----------
    fold : dict
        Output of ``build_loo_folds`` — contains *fold_name*,
        *train_files*, *test_healthy_file*, *test_tumor_files*.
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    dict
        Keys: fold_name, held_out_patient, tumor_patient, patient_auc,
        seq_auc, num_train_seqs, num_test_seqs, per_patient_data.
    """
    fold_name = fold["fold_name"]
    logger.info("")
    logger.info("=" * 60)
    logger.info(f" FOLD: {fold_name}")
    logger.info("=" * 60)
    logger.info(f"  Training on   : {[os.path.basename(f) for f in fold['train_files']]}")
    logger.info(f"  Held-out test : {os.path.basename(fold['test_healthy_file'])}")
    logger.info(f"  Tumor test    : {[os.path.basename(f) for f in fold['test_tumor_files']]}")

    # --- Load data ---
    train_data, test_data, y_test_true_seq, test_files_info = load_tracked_patient_cohort(
        fold["train_files"],
        [fold["test_healthy_file"]],
        fold["test_tumor_files"],
        args.max_train,
        args.max_test,
        args.max_test,
        args.seed,
        args.cache_dir,
    )

    # --- Kernel computation (train) ---
    mkl_weights = generate_mkl_weights(
        args.max_k, noise_threshold=max(1, 2 * args.mismatches)
    )
    logger.info(
        f"Computing Mismatch Kernel (Max K: {args.max_k}, "
        f"Mismatches: {args.mismatches}) ..."
    )

    K_train_unnorm, train_states = mixed_string_kernel(
        sequences=train_data,
        k_max=args.max_k,
        m=args.mismatches,
        weights=mkl_weights,
        n_jobs=args.n_jobs,
    )
    K_train = normalize_gram(K_train_unnorm)

    # --- Kernel computation (test, asymmetric) ---
    logger.info("Computing Asymmetric Kernel for Testing ...")
    K_test = compute_asymmetric_normalized_kernel(
        test_seqs=test_data,
        train_states=train_states,
        max_k=args.max_k,
        mismatches=args.mismatches,
        mkl_weights=mkl_weights,
        n_jobs=args.n_jobs,
    )

    # --- Sequence-level evaluation ---
    logger.info(f"Fitting One-Class SVM (nu={args.nu_param}) ...")
    metrics = evaluate_novelty_detector(
        K_train=K_train,
        K_test=K_test,
        y_test_true=y_test_true_seq,
        nu=args.nu_param,
    )

    # --- Patient-level aggregation ---
    patient_auc, per_patient_data = evaluate_patient_level_novelty(
        metrics["anomaly_scores"],
        test_files_info,
    )

    # --- Per-fold plot (organized by tested tumor subject) ---
    if args.plot_dir:
        from src.plotting import generate_loo_fold_plot

        tumor_base = os.path.basename(fold["test_tumor_files"][0])
        tumor_subject = "_".join(tumor_base.split("_")[:2])  # e.g. Colo_11
        fold_plot_dir = os.path.join(
            args.plot_dir,
            f"m_{args.mismatches}",
            f"k_{args.max_k}",
            tumor_subject,
        )
        generate_loo_fold_plot(per_patient_data, fold_plot_dir, fold_name, args.seed)

    # --- Derive short tumor patient name ---
    tumor_base = os.path.basename(fold["test_tumor_files"][0])
    tumor_tag = "_".join(tumor_base.split("_")[:2])  # e.g. Colo_11

    logger.info(f"[{fold_name}] Patient-Level AUC: {patient_auc:.4f}  |  Seq-Level AUC: {metrics['auc']:.4f}")

    # Extract per-patient mean anomaly scores
    mean_healthy = next(d['mean_score'] for d in per_patient_data if d['label'] != -1)
    mean_tumor = next(d['mean_score'] for d in per_patient_data if d['label'] == -1)

    return {
        "fold_name": fold_name,
        "held_out_patient": fold_name.replace("LOO_", ""),
        "tumor_patient": tumor_tag,
        "patient_auc": patient_auc,
        "seq_auc": metrics["auc"],
        "mean_score_healthy": mean_healthy,
        "mean_score_tumor": mean_tumor,
        "num_train_seqs": len(train_data),
        "num_test_seqs": len(test_data),
        "per_patient_data": per_patient_data,
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = create_base_parser(
        "Leave-One-Out Cross-Validation over Healthy Patients (String-Kernel MIL)"
    )
    add_data_dir_arg(parser, required=True)
    add_cache_dir_arg(parser, project_root)
    add_train_sampling_arg(parser)
    parser.add_argument(
        "--max-test", type=int, default=1500,
        help="Max sequences to sample from each test patient — healthy and tumor (default: 1500).",
    )
    add_seed_arg(parser)
    add_kernel_args(parser)
    add_nu_arg(parser)
    add_execution_args(parser)
    add_loo_args(parser)
    args = parser.parse_args()

    # Default training budget: 30 000 (5 patients × 6 000 each)
    # Override only if the user did not explicitly set --max-train
    # (argparse default is 18 000; we bump it here for LOO)
    if args.max_train == 18000:
        args.max_train = 30000
        logger.info("Auto-increased --max-train to 30000 for 5-patient LOO folds")

    logger.info("=" * 60)
    logger.info(" COLON CANCER SOMATIC DETECTION")
    logger.info(" LEAVE-ONE-OUT CROSS-VALIDATION (STRING-KERNEL MIL)")
    logger.info("=" * 60)

    # Validate the tumor file exists
    if not validate_files_exist([args.tumor_file], logger):
        sys.exit(1)

    # Derive tested tumor subject name for organizing outputs
    tumor_base = os.path.basename(args.tumor_file)
    tumor_subject = "_".join(tumor_base.split("_")[:2])  # e.g. "Colo_11"

    # Build the LOO folds
    folds = build_loo_folds(args.data_dir, args.tumor_file)
    logger.info(f"Generated {len(folds)} LOO folds over healthy patients")

    # Validate ALL fold files exist up-front
    all_files = set()
    for fold in folds:
        all_files.update(fold["train_files"])
        all_files.add(fold["test_healthy_file"])
        all_files.update(fold["test_tumor_files"])
    if not validate_files_exist(list(all_files), logger):
        sys.exit(1)

    # --- Run each fold ---
    start_time = time.time()
    results = []
    for fold in folds:
        result = _run_single_fold(fold, args)
        results.append(result)

    elapsed = time.time() - start_time

    # --- Aggregate & report ---
    print_loo_summary(results, logger)
    logger.info(f"Total LOO Execution Time: {elapsed:.2f} seconds")

    # --- Save CSV into subject subfolder ---
    csv_path = args.output_csv
    if csv_path is None:
        subject_dir = os.path.join(
            project_root, "results", "loo",
            f"m_{args.mismatches}", f"k_{args.max_k}", tumor_subject,
        )
        os.makedirs(subject_dir, exist_ok=True)
        csv_path = os.path.join(subject_dir, "loo_results.csv")
    save_loo_results(results, csv_path)


if __name__ == "__main__":
    main()

