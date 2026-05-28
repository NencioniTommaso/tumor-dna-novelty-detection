"""
validate_reference.py
Validates that the reference distribution is stable regardless of which
subset of healthy patients is used.

Randomly samples N combinations of 3 healthy files (out of 6 available),
computes the reference KDE for each, and produces:
  1. An overlay plot of all reference KDE curves
  2. A pairwise OA heatmap between all combinations
  3. Summary statistics (mean, min, std of pairwise OAs)
  4. A JSON file with machine-readable results
"""

import json
import random
import sys
import os
import time
from itertools import combinations

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.data_utils import load_train_cohort_only
from src.kernels import generate_mkl_weights, mixed_string_kernel, normalize_gram
from src.evaluation_oa import compute_reference_kde, compute_oa
from src.plotter_oa import (
    init_overlapping_plot,
    add_overlapping_curves,
    plot_pairwise_oa_heatmap,
)
from experiments.experiments_utils import (
    setup_logger,
    create_base_parser,
    add_data_dir_arg,
    add_cache_dir_arg,
    add_train_sampling_arg,
    add_seed_arg,
    add_kernel_args,
    add_execution_args,
    build_all_healthy_files,
    validate_files_exist,
)

logger = setup_logger(__name__)


def _short_name(filepath):
    """Extract a short patient label from a FASTA path, e.g. 'H5'."""
    basename = os.path.basename(filepath)
    parts = basename.split("_")
    if len(parts) >= 2:
        return f"H{parts[1]}"
    return basename


def main():
    parser = create_base_parser(
        "Validate reference distribution stability across different healthy patient combinations"
    )
    add_data_dir_arg(parser, required=True)
    add_cache_dir_arg(parser, project_root)
    add_train_sampling_arg(parser)
    add_seed_arg(parser)
    add_kernel_args(parser)
    add_execution_args(parser)
    parser.add_argument(
        "--n-combos",
        type=int,
        default=3,
        help="Number of random combinations of 3 healthy files to test (default: 3).",
    )
    args = parser.parse_args()

    logger.info("=====================================================")
    logger.info(" REFERENCE DISTRIBUTION STABILITY VALIDATION")
    logger.info("=====================================================")

    # --- 1. Enumerate and sample combinations ---
    all_healthy = build_all_healthy_files(args.data_dir)
    if not validate_files_exist(all_healthy, logger):
        sys.exit(1)

    all_combos = list(combinations(all_healthy, 3))
    n_combos = min(args.n_combos, len(all_combos))

    random.seed(args.seed)
    selected_combos = random.sample(all_combos, n_combos)

    logger.info(f"Total healthy files: {len(all_healthy)}")
    logger.info(f"Possible combinations of 3: {len(all_combos)}")
    logger.info(f"Selected combinations: {n_combos}")

    # --- 2. Compute reference KDE for each combination ---
    mkl_weights = generate_mkl_weights(args.max_k, noise_threshold=max(1, 2 * args.mismatches))
    downsample_kde = not args.disable_kde_downsampling

    ref_kdes = []  # list of (label, xs, y_intra)
    start_time = time.perf_counter()

    for idx, combo in enumerate(selected_combos):
        label = "+".join(_short_name(f) for f in combo)
        logger.info(f"\nProcessing combination {idx + 1}/{n_combos}: {label}")

        # Load training data for this combination
        train_data = load_train_cohort_only(
            list(combo), args.max_train, args.seed, args.cache_dir, logger
        )

        # Compute kernel
        K_train_unnorm, _ = mixed_string_kernel(
            sequences=train_data,
            k_max=args.max_k,
            m=args.mismatches,
            weights=mkl_weights,
            n_jobs=args.n_jobs,
        )
        K_train = normalize_gram(K_train_unnorm)

        # Compute reference KDE
        xs, y_intra = compute_reference_kde(K_train, downsample_kde=downsample_kde)
        ref_kdes.append((label, xs, y_intra))
        logger.info(f"  Done: {label}")

    elapsed = time.perf_counter() - start_time
    logger.info(f"\nAll {n_combos} combinations computed in {elapsed:.2f} seconds")

    # --- 3. Overlay plot ---
    plot_dir = args.plot_dir
    if plot_dir is None:
        plot_dir = os.path.join(project_root, "results", "reference_validation")

    os.makedirs(plot_dir, exist_ok=True)

    metric_name = f"kernel (m={args.mismatches}, k={args.max_k})"
    first_label, first_xs, first_y = ref_kdes[0]

    fig, ax = init_overlapping_plot(
        first_xs,
        first_y,
        y_inter=None,
        metric_name=metric_name,
        sample_size=args.max_train,
        label1=first_label,
        title="Reference Distributions — Stability Check",
        plt_overlap=False,
        line_width=1.5,
    )

    for label, xs, y_intra in ref_kdes[1:]:
        add_overlapping_curves(ax, xs, y_intra, label=label, line_width=1.5)

    ax.legend(loc="upper right", frameon=False, fontsize=7)
    fig.tight_layout()

    import matplotlib.pyplot as plt
    overlay_path = os.path.join(plot_dir, f"reference_overlay_seed{args.seed}.pdf")
    fig.savefig(overlay_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved overlay plot: {overlay_path}")

    # --- 4. Pairwise OA matrix ---
    n = len(ref_kdes)
    oa_matrix = np.ones((n, n))
    labels = [label for label, _, _ in ref_kdes]

    for i in range(n):
        for j in range(i + 1, n):
            _, xs_i, y_i = ref_kdes[i]
            _, xs_j, y_j = ref_kdes[j]
            oa = compute_oa(y_i, y_j, xs_i)
            oa_matrix[i, j] = oa
            oa_matrix[j, i] = oa

    # Extract upper triangle (excluding diagonal) for statistics
    triu_indices = np.triu_indices(n, k=1)
    pairwise_oas = oa_matrix[triu_indices]

    logger.info("\n--- Pairwise OA Summary ---")
    logger.info(f"Mean pairwise OA: {pairwise_oas.mean():.4f}")
    logger.info(f"Min  pairwise OA: {pairwise_oas.min():.4f}")
    logger.info(f"Max  pairwise OA: {pairwise_oas.max():.4f}")
    logger.info(f"Std  pairwise OA: {pairwise_oas.std():.4f}")

    # Heatmap plot
    heatmap_path = plot_pairwise_oa_heatmap(oa_matrix, labels, plot_dir, args.seed)
    logger.info(f"Saved heatmap: {heatmap_path}")

    # --- 5. Save JSON results ---
    results = {
        "seed": args.seed,
        "max_train": args.max_train,
        "max_k": args.max_k,
        "mismatches": args.mismatches,
        "n_combos": n_combos,
        "labels": labels,
        "oa_matrix": oa_matrix.tolist(),
        "summary": {
            "mean": float(pairwise_oas.mean()),
            "min": float(pairwise_oas.min()),
            "max": float(pairwise_oas.max()),
            "std": float(pairwise_oas.std()),
        },
    }

    json_path = os.path.join(plot_dir, f"reference_validation_seed{args.seed}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved JSON results: {json_path}")

    logger.info("\n=====================================================")
    logger.info(" VALIDATION COMPLETE")
    logger.info("=====================================================")


if __name__ == "__main__":
    main()
