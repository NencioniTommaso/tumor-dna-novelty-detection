"""
run_train_seed_stability.py
Train-Seed Stability Experiment for the String-Kernel MIL pipeline.

Measures how sensitive the OA (Overlapping Area) between Healthy_3 and
Healthy_4 anomaly-score distributions is when re-sampling the training set
from Healthy_2 with different random seeds.

Protocol
--------
  1. Load 10k sequences from H3 and 10k from H4 with a *fixed* seed (42).
  2. For each of 10 training seeds (42 … 51):
       a. Sample 30k sequences from H2.
       b. Compute the string-kernel Gram matrix and train an OCSVM.
       c. Score H3 and H4 through the OCSVM (anomaly scores).
       d. Compute OA between KDE(H3 scores) vs KDE(H4 scores).
       e. Save the raw anomaly scores to a per-seed CSV.
  3. Save an aggregated results CSV and print summary statistics.
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

from src.fasta_reader import MMapFastaReader
from src.gram import (
    generate_mkl_weights,
    mixed_string_kernel,
    normalize_gram,
    compute_asymmetric_normalized_kernel,
)
from src.loo_results import save_fold_anomaly_scores
from experiments.compute_oa_from_scores import compute_oa_for_pair
from experiments.experiments_utils import (
    setup_logger,
    create_base_parser,
    add_data_dir_arg,
    add_cache_dir_arg,
    add_kernel_args,
    add_nu_arg,
    add_execution_args,
    validate_files_exist,
)

logger = setup_logger(__name__)

# ── Default experiment parameters ──────────────────────────────────────
NUM_RUNS = 10
BASE_SEED = 42          # first training seed; also used for fixed test sampling
TEST_SEED = 42          # seed for H3/H4 sampling (stays fixed)
DEFAULT_MAX_TRAIN = 30_000
DEFAULT_MAX_TEST = 10_000


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _load_seqs(fasta_path: str, n_seqs: int, seed: int, cache_dir: str) -> list[str]:
    """Sample *n_seqs* sequences from *fasta_path* using *seed*."""
    np.random.seed(seed)
    reader = MMapFastaReader(fasta_path, index_cache_dir=cache_dir)
    total = len(reader.offsets)
    n = min(n_seqs, total)
    indices = np.random.choice(total, n, replace=False)
    seqs = [reader.get_seq(i).upper() for i in indices]
    reader.close()
    return seqs





# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = create_base_parser(
        "Train-Seed Stability: OA sensitivity to H2 training subsamples"
    )
    add_data_dir_arg(parser, required=True)
    add_cache_dir_arg(parser, project_root)
    add_kernel_args(parser)
    add_nu_arg(parser)
    add_execution_args(parser)
    parser.add_argument(
        "--max-train", type=int, default=DEFAULT_MAX_TRAIN,
        help=f"Sequences to sample from H2 per run (default: {DEFAULT_MAX_TRAIN}).",
    )
    parser.add_argument(
        "--max-test", type=int, default=DEFAULT_MAX_TEST,
        help=f"Sequences to sample from each of H3/H4 (default: {DEFAULT_MAX_TEST}).",
    )
    parser.add_argument(
        "--num-runs", type=int, default=NUM_RUNS,
        help=f"Number of runs with different training seeds (default: {NUM_RUNS}).",
    )
    parser.add_argument(
        "--base-seed", type=int, default=BASE_SEED,
        help=f"First training seed; seeds will be base_seed … base_seed+num_runs-1 (default: {BASE_SEED}).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for results (default: results/stability/m_<m>/k_<k>/).",
    )
    args = parser.parse_args()

    # ── Resolve file paths ────────────────────────────────────────────
    h2_path = os.path.join(args.data_dir, "Healthy_2_merged_subset_1200000.fa")
    h3_path = os.path.join(args.data_dir, "Healthy_3_merged_subset_1200000.fa")
    h4_path = os.path.join(args.data_dir, "Healthy_4_merged_subset_1200000.fa")

    if not validate_files_exist([h2_path, h3_path, h4_path], logger):
        sys.exit(1)

    # ── Output directory ──────────────────────────────────────────────
    out_dir = args.output_dir or os.path.join(
        project_root, "results", "stability",
        f"m_{args.mismatches}", f"k_{args.max_k}",
    )
    scores_dir = os.path.join(out_dir, "anomaly_scores")
    os.makedirs(scores_dir, exist_ok=True)

    # ── Banner ────────────────────────────────────────────────────────
    train_seeds = list(range(args.base_seed, args.base_seed + args.num_runs))

    logger.info("=" * 65)
    logger.info(" TRAIN-SEED STABILITY EXPERIMENT")
    logger.info("=" * 65)
    logger.info(f"  Training patient : Healthy_2  ({args.max_train} seqs/run)")
    logger.info(f"  Test patients    : Healthy_3 + Healthy_4  ({args.max_test} seqs each)")
    logger.info(f"  Test seed (fixed): {TEST_SEED}")
    logger.info(f"  Training seeds   : {train_seeds}")
    logger.info(f"  Kernel           : max_k={args.max_k}, m={args.mismatches}")
    logger.info(f"  OCSVM nu         : {args.nu_param}")
    logger.info(f"  Output dir       : {out_dir}")
    logger.info("=" * 65)

    # ── Step 1: Load test data ONCE (fixed seed) ──────────────────────
    logger.info("")
    logger.info("Loading test data (fixed seed=%d) ...", TEST_SEED)
    test_h3 = _load_seqs(h3_path, args.max_test, TEST_SEED, args.cache_dir)
    # We need H4 sampling to be independent of H3 but still deterministic.
    # After H3 sampling consumed some RNG state, we re-seed for H4.
    test_h4 = _load_seqs(h4_path, args.max_test, TEST_SEED, args.cache_dir)
    test_data = test_h3 + test_h4
    n_h3, n_h4 = len(test_h3), len(test_h4)
    logger.info(f"  Healthy_3: {n_h3} seqs  |  Healthy_4: {n_h4} seqs")

    # ── Kernel parameters ─────────────────────────────────────────────
    mkl_weights = generate_mkl_weights(
        args.max_k, noise_threshold=max(1, 2 * args.mismatches)
    )

    # ── Step 2: Loop over training seeds ──────────────────────────────
    results = []
    total_start = time.time()

    for run_idx, train_seed in enumerate(train_seeds):
        run_start = time.time()
        logger.info("")
        logger.info("-" * 65)
        logger.info(f" RUN {run_idx + 1}/{len(train_seeds)}  |  train_seed = {train_seed}")
        logger.info("-" * 65)

        # 2a. Load training data with current seed
        train_data = _load_seqs(h2_path, args.max_train, train_seed, args.cache_dir)
        logger.info(f"  Loaded {len(train_data)} training seqs from Healthy_2")

        # 2b. Training kernel
        logger.info("  Computing training Gram matrix ...")
        K_train_unnorm, train_states = mixed_string_kernel(
            sequences=train_data,
            k_max=args.max_k,
            m=args.mismatches,
            weights=mkl_weights,
            n_jobs=args.n_jobs,
        )
        K_train = normalize_gram(K_train_unnorm)
        del K_train_unnorm  # free ~7 GB immediately

        # 2c. Asymmetric kernel (test vs train)
        logger.info("  Computing asymmetric test kernel ...")
        K_test = compute_asymmetric_normalized_kernel(
            test_seqs=test_data,
            train_states=train_states,
            max_k=args.max_k,
            mismatches=args.mismatches,
            mkl_weights=mkl_weights,
            n_jobs=args.n_jobs,
        )
        del train_states  # sparse feature matrices no longer needed

        # 2d. Fit OCSVM & score
        logger.info(f"  Fitting OCSVM (nu={args.nu_param}) ...")
        ocsvm = OneClassSVM(kernel="precomputed", nu=args.nu_param)
        ocsvm.fit(K_train)
        anomaly_scores = ocsvm.decision_function(K_test)

        scores_h3 = anomaly_scores[:n_h3]
        scores_h4 = anomaly_scores[n_h3:]

        # Invert so higher = more anomalous (consistent with rest of codebase)
        inv_h3 = -scores_h3
        inv_h4 = -scores_h4

        # 2e. OA between H3 and H4 score distributions
        #     Uses the same compute_oa_for_pair as compute_oa_from_scores.py
        #     (shared KDE grid, _fit_kde, OA = 1 - 0.5 * ∫|f_a - f_b|dx)
        oa = compute_oa_for_pair(inv_h3, inv_h4)
        mean_h3 = float(np.mean(inv_h3))
        mean_h4 = float(np.mean(inv_h4))

        run_elapsed = time.time() - run_start
        logger.info(f"  OA(H3 vs H4) = {oa:.4f}  |  mean_h3={mean_h3:.4f}  mean_h4={mean_h4:.4f}  [{run_elapsed:.1f}s]")

        results.append({
            "train_seed": train_seed,
            "oa": oa,
            "mean_score_h3": mean_h3,
            "mean_score_h4": mean_h4,
            "n_train": len(train_data),
            "n_test_h3": n_h3,
            "n_test_h4": n_h4,
        })

        # 2f. Save raw anomaly scores (same format as LOO experiment)
        per_patient_data = [
            {"short_name": "Healthy_3", "inverted_scores": inv_h3},
            {"short_name": "Healthy_4", "inverted_scores": inv_h4},
        ]
        scores_path = os.path.join(scores_dir, f"seed{train_seed}_scores.csv")
        save_fold_anomaly_scores(per_patient_data, scores_path)

        # ── Free all large objects before next iteration ──
        del train_data, K_train, K_test, ocsvm, anomaly_scores
        del scores_h3, scores_h4, inv_h3, inv_h4, per_patient_data
        gc.collect()

    total_elapsed = time.time() - total_start

    # ── Step 3: Aggregate & save ──────────────────────────────────────
    oas = np.array([r["oa"] for r in results])

    logger.info("")
    logger.info("=" * 65)
    logger.info(" SUMMARY")
    logger.info("=" * 65)
    logger.info(f"  Runs             : {len(results)}")
    logger.info(f"  OA mean ± std    : {np.mean(oas):.4f} ± {np.std(oas):.4f}")
    logger.info(f"  OA min / max     : {np.min(oas):.4f} / {np.max(oas):.4f}")
    logger.info(f"  Total time       : {total_elapsed:.1f}s")
    logger.info("=" * 65)

    # Save aggregated CSV
    csv_path = os.path.join(out_dir, "stability_results.csv")
    fieldnames = ["train_seed", "oa", "mean_score_h3", "mean_score_h4",
                  "n_train", "n_test_h3", "n_test_h4"]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    logger.info(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
