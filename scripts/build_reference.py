"""
build_reference.py
Computes a reference distribution from healthy baseline data and persists
it to disk for reuse across multiple experiment runs.

The reference artifact contains the normalized training Gram matrix,
per-k train_states (for asymmetric kernel computation at test time),
and the intra-distance KDE (xs, y_intra).
"""

import sys
import os
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.data_utils import load_train_cohort_only
from src.kernels import generate_mkl_weights, mixed_string_kernel, normalize_gram
from src.evaluation_oa import compute_reference_kde
from src.reference_io import save_reference
from experiments.experiments_utils import (
    setup_logger,
    create_base_parser,
    add_data_dir_arg,
    add_cache_dir_arg,
    add_train_sampling_arg,
    add_seed_arg,
    add_kernel_args,
    add_execution_args,
    build_train_normal_files,
    validate_files_exist,
)

logger = setup_logger(__name__)


def main():
    parser = create_base_parser("Build and save a reference distribution from healthy baseline data")
    add_data_dir_arg(parser, required=True)
    add_cache_dir_arg(parser, project_root)
    add_train_sampling_arg(parser)
    add_seed_arg(parser)
    add_kernel_args(parser)
    add_execution_args(parser)
    parser.add_argument(
        "--ref-name",
        type=str,
        default="reference.pkl",
        help="Filename for the saved reference artifact (default: reference.pkl).",
    )
    parser.add_argument(
        "--train-files",
        nargs="+",
        type=str,
        default=None,
        help="Optional: explicit list of healthy FASTA files to use. "
             "If not set, uses the default training split (Healthy_2..4).",
    )
    args = parser.parse_args()

    logger.info("=====================================================")
    logger.info(" BUILD REFERENCE DISTRIBUTION")
    logger.info("=====================================================")

    # --- 1. Determine which files to use ---
    if args.train_files:
        train_normal_files = args.train_files
    else:
        train_normal_files = build_train_normal_files(args.data_dir)

    if not validate_files_exist(train_normal_files, logger):
        sys.exit(1)

    logger.info(f"Reference files: {[os.path.basename(f) for f in train_normal_files]}")
    logger.info(f"Seed: {args.seed} | Max train: {args.max_train} | max_k: {args.max_k} | m: {args.mismatches}")

    # --- 2. Load training data ---
    train_data = load_train_cohort_only(
        train_normal_files, args.max_train, args.seed, args.cache_dir, logger
    )

    start_time = time.perf_counter()

    # --- 3. Compute Kernel ---
    mkl_weights = generate_mkl_weights(args.max_k, noise_threshold=max(1, 2 * args.mismatches))
    logger.info(f"\nComputing Mixed String Kernel (Train x Train)...")

    K_train_unnorm, train_states = mixed_string_kernel(
        sequences=train_data,
        k_max=args.max_k,
        m=args.mismatches,
        weights=mkl_weights,
        n_jobs=args.n_jobs,
    )

    logger.info("Normalizing Training Gram Matrix...")
    K_train = normalize_gram(K_train_unnorm)

    # --- 4. Compute Reference KDE ---
    logger.info("Computing reference KDE (intra-distances)...")
    downsample_kde = not args.disable_kde_downsampling
    xs, y_intra = compute_reference_kde(K_train, downsample_kde=downsample_kde)

    # --- 5. Save the reference artifact ---
    save_dir = os.path.join(project_root, "models", "references")
    save_path = os.path.join(save_dir, args.ref_name)

    state = {
        'ref_seed': args.seed,
        'train_sequences': train_data,
        'K_train': K_train,
        'train_states': train_states,
        'xs': xs,
        'y_intra': y_intra,
        'max_k': args.max_k,
        'mismatches': args.mismatches,
        'mkl_weights': mkl_weights,
        'max_train': args.max_train,
        'train_files': train_normal_files,
    }

    save_reference(state, save_path)

    elapsed = time.perf_counter() - start_time
    logger.info(f"\nReference build time: {elapsed:.2f} seconds")
    logger.info(f"Artifact saved to: {save_path}")
    logger.info("Reference build complete.")


if __name__ == "__main__":
    main()
