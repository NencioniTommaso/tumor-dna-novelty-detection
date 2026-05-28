"""
run_mil_experiment_oa.py
Executes a Patient-Level MIL pipeline for Colon Cancer Novelty Detection
using the Overlapping Area (OA) KDE methodology.

Supports two modes:
  1. Standard mode (no --ref-path): computes everything from scratch.
  2. Fixed-reference mode (--ref-path): loads a pre-built reference artifact
     and only samples/computes test data. The --seed only controls test sampling.
"""

import time
import sys
import os

# Dynamically resolve paths to ensure the script runs from anywhere
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Import from our custom library
from src.data_utils import load_tracked_patient_cohort, load_test_cohort_only
from src.kernels import generate_mkl_weights, mixed_string_kernel, normalize_gram, compute_asymmetric_normalized_kernel
from src.evaluation_oa import evaluate_patient_level_oa_method
from experiments.experiments_utils import (
    setup_logger,
    create_base_parser,
    add_data_dir_arg,
    add_cache_dir_arg,
    add_train_sampling_arg,
    add_test_sampling_args,
    add_seed_arg,
    add_kernel_args,
    add_execution_args,
    add_ref_path_arg,
    build_default_cohorts,
    build_test_normal_files,
    build_tumor_files,
    validate_files_exist,
)

logger = setup_logger(__name__)


def main():
    parser = create_base_parser("Run Sequence Novelty Detection Experiments (OA KDE Method)")
    add_data_dir_arg(parser, required=True)
    add_cache_dir_arg(parser, project_root)
    add_train_sampling_arg(parser)
    add_test_sampling_args(parser)
    add_seed_arg(parser)
    add_kernel_args(parser)
    add_execution_args(parser)
    add_ref_path_arg(parser)
    args = parser.parse_args()
    
    logger.info("=====================================================")
    logger.info(" COLON CANCER SOMATIC DETECTION: MULTIPLE INSTANCE LEARNING (OA METHOD)")
    logger.info("=====================================================")

    if args.ref_path:
        # ===== FIXED-REFERENCE MODE =====
        from src.reference_io import load_reference

        ref = load_reference(args.ref_path)
        K_train = ref['K_train']
        train_states = ref['train_states']
        mkl_weights = ref['mkl_weights']
        xs_ref = ref['xs']
        y_intra_ref = ref['y_intra']
        max_k = ref['max_k']
        mismatches = ref['mismatches']

        logger.info(f"Using fixed reference (seed={ref['ref_seed']}). Test seed={args.seed}")

        # Load ONLY test data with the experiment seed
        test_normal_files = build_test_normal_files(args.data_dir)
        test_tumor_files = build_tumor_files(args.data_dir)

        test_files = test_normal_files + test_tumor_files
        if not validate_files_exist(test_files, logger):
            sys.exit(1)

        test_data, y_test_true_seq, test_files_info = load_test_cohort_only(
            test_normal_files,
            test_tumor_files,
            args.max_test_normal,
            args.max_test_tumor,
            args.seed,
            args.cache_dir,
            logger,
        )

        # Compute asymmetric kernel (test vs fixed train)
        start_time = time.time()
        logger.info("\nComputing Asymmetric Kernel for Testing...")
        K_test = compute_asymmetric_normalized_kernel(
            test_seqs=test_data,
            train_states=train_states,
            max_k=max_k,
            mismatches=mismatches,
            mkl_weights=mkl_weights,
            n_jobs=args.n_jobs,
        )

        # Evaluate with pre-computed reference
        final_plot_dir = None
        if args.plot_dir:
            final_plot_dir = os.path.join(args.plot_dir, f"m_{mismatches}", f"k_{max_k}")

        patient_auc = evaluate_patient_level_oa_method(
            K_train, K_test, test_files_info, logger,
            n_jobs=args.n_jobs,
            downsample_kde=not args.disable_kde_downsampling,
            plot_dir=final_plot_dir,
            mismatches=mismatches,
            max_k=max_k,
            seed=args.seed,
            xs_ref=xs_ref,
            y_intra_ref=y_intra_ref,
        )

    else:
        # ===== STANDARD MODE (backward compatible) =====

        # --- 1. Define the Patient-Level Split ---
        train_normal_files, test_normal_files, test_tumor_files = build_default_cohorts(args.data_dir)

        # Verify files exist before running
        all_files = train_normal_files + test_normal_files + test_tumor_files
        if not validate_files_exist(all_files, logger):
            sys.exit(1)

        # --- 2. Load and Sample Tracked Data  ---
        logger.info("\nStarting data loading and tracking...")
        train_data, test_data, y_test_true_seq, test_files_info = load_tracked_patient_cohort(
            train_normal_files, 
            test_normal_files, 
            test_tumor_files,
            args.max_train,
            args.max_test_normal,
            args.max_test_tumor,
            args.seed,
            args.cache_dir,
            logger
        )
        
        # --- 3. Kernel Computation (Train) ---
        mkl_weights = generate_mkl_weights(args.max_k, noise_threshold=max(1, 2 * args.mismatches))
        logger.info(f"\nComputing Explicit Sparse Mismatch Kernel for Training (Max K: {args.max_k}, Mismatches: {args.mismatches})...")
        
        start_time = time.time()
        K_train_unnorm, train_states = mixed_string_kernel(
            sequences=train_data, 
            k_max=args.max_k, 
            m=args.mismatches, 
            weights=mkl_weights,
            n_jobs=args.n_jobs  
        )
        
        logger.info("Normalizing Training Gram Matrix...")
        K_train = normalize_gram(K_train_unnorm)
        
        # --- 4. Kernel Computation (Test Asymmetric) ---
        logger.info("\nComputing Asymmetric Kernel for Testing...")
        K_test = compute_asymmetric_normalized_kernel(
            test_seqs=test_data,
            train_states=train_states,
            max_k=args.max_k,
            mismatches=args.mismatches,
            mkl_weights=mkl_weights,
            n_jobs=args.n_jobs
        )
        
        # --- 5. True Patient-Level Anomaly Aggregation via KDE & OA ---
        final_plot_dir = None
        if args.plot_dir:
            final_plot_dir = os.path.join(args.plot_dir, f"m_{args.mismatches}", f"k_{args.max_k}")

        patient_auc = evaluate_patient_level_oa_method(
            K_train, K_test, test_files_info, logger, 
            n_jobs=args.n_jobs, 
            downsample_kde=not args.disable_kde_downsampling,
            plot_dir=final_plot_dir,
            mismatches=args.mismatches,
            max_k=args.max_k,
            seed=args.seed
        )
    
    elapsed = time.time() - start_time
    
    # --- Output Results ---
    logger.info("\n=====================================================")
    logger.info(" FINAL RESULTS: PATIENT-LEVEL OA EVALUATION")
    logger.info("=====================================================")
    logger.info(f"Execution Time          : {elapsed:.2f} seconds")
    logger.info(f"PATIENT-LEVEL ROC-AUC   : {patient_auc:.4f} (True Clinical Metric)")
    logger.info("=====================================================")

if __name__ == "__main__":
    main()
