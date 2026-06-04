"""
run_deep_mil_experiment_oa.py
Executes a Deep Learning-Based Patient-Level Multiple Instance Learning (MIL)
pipeline for Colon Cancer Novelty Detection using the Overlapping Area (OA) KDE methodology.

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
from src.evaluation_oa import evaluate_patient_level_oa_method
from experiments.experiments_utils import (
    setup_logger,
    create_base_parser,
    add_data_dir_arg,
    add_cache_dir_arg,
    add_train_sampling_arg,
    add_test_sampling_args,
    add_seed_arg,
    add_execution_args,
    add_ref_path_arg,
    build_default_cohorts,
    build_test_normal_files,
    build_tumor_files,
    validate_files_exist,
)

logger = setup_logger(__name__)


def main():
    parser = create_base_parser("Run Deep Sequence Novelty Detection Experiments (OA KDE Method)")
    add_data_dir_arg(parser, required=True)
    add_cache_dir_arg(parser, project_root)
    add_train_sampling_arg(parser)
    add_test_sampling_args(parser)
    add_seed_arg(parser)
    add_execution_args(parser)
    add_ref_path_arg(parser)
    args = parser.parse_args()
    
    logger.info("=====================================================")
    logger.info(" COLON CANCER SOMATIC DETECTION: DEEP LEARNING MIL (OA METHOD)")
    logger.info("=====================================================")

    if args.ref_path:
        # ===== FIXED-REFERENCE MODE =====
        from src.reference_io import load_reference

        ref = load_reference(args.ref_path)
        K_train = ref['K_train']
        xs_ref = ref['xs']
        y_intra_ref = ref['y_intra']

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

        # Deep Kernel Computation
        start_time = time.time()

        # Lazy import to avoid slow PyTorch/Transformers initialization during --help
        from src.DNAFeatureExtractor import compute_train_test_kernels

        logger.info("\nComputing Deep Kernels using Foundation Model...")
        # Need the train sequences from the reference for the deep kernel
        train_data = ref.get('train_sequences', [])
        if not train_data:
            logger.error("Reference artifact does not contain train_sequences. "
                         "Cannot compute deep kernel in fixed-reference mode.")
            sys.exit(1)

        _, K_test = compute_train_test_kernels(
            train_sequences=train_data,
            test_sequences=test_data,
            model_name="quietflamingo/dnabert2-no-flashattention",
            kernel_type="rbf",
            batch_size=8,
        )

        # Evaluate with pre-computed reference
        final_plot_dir = None
        if args.plot_dir:
            final_plot_dir = os.path.join(args.plot_dir, "deep")

        patient_auc = evaluate_patient_level_oa_method(
            K_train, K_test, test_files_info, logger,
            n_jobs=args.n_jobs,
            downsample_kde=not args.disable_kde_downsampling,
            plot_dir=final_plot_dir,
            is_deep=True,
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

        # --- 2. Load and Sample Tracked Data ---
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
        
        # --- 3. Deep Kernel Computation ---
        start_time = time.time()
        
        # Lazy import to avoid slow PyTorch/Transformers initialization during --help
        from src.DNAFeatureExtractor import compute_train_test_kernels

        logger.info("\nComputing Deep Kernels using Foundation Model...")
        # K_train is (Train vs Train), K_test is (Test vs Train)
        K_train, K_test = compute_train_test_kernels(
            train_sequences=train_data,
            test_sequences=test_data,
            model_name="quietflamingo/dnabert2-no-flashattention", 
            kernel_type="rbf",
            batch_size=8
        )
        
        # --- 4. True Patient-Level Anomaly Aggregation via KDE & OA ---
        final_plot_dir = None
        if args.plot_dir:
            final_plot_dir = os.path.join(args.plot_dir, "deep")

        patient_auc = evaluate_patient_level_oa_method(
            K_train, K_test, test_files_info, logger, 
            n_jobs=args.n_jobs, 
            downsample_kde=not args.disable_kde_downsampling,
            plot_dir=final_plot_dir,
            is_deep=True,
            seed=args.seed
        )
    
    elapsed = time.time() - start_time
    
    # --- Output Results ---
    logger.info("\n=====================================================")
    logger.info(" FINAL RESULTS: PATIENT-LEVEL DEEP OA EVALUATION")
    logger.info("=====================================================")
    logger.info(f"Execution Time          : {elapsed:.2f} seconds")
    logger.info(f"PATIENT-LEVEL ROC-AUC   : {patient_auc:.4f} (True Clinical Metric)")
    logger.info("=====================================================")

if __name__ == "__main__":
    main()
