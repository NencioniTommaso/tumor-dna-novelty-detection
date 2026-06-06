"""
run_full_inference.py
Runs full cohort inference using a saved SVM model to replicate exact MIL
experiment sequences. Outputs raw anomaly scores and patient-level AUC.
"""

import sys
import os
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from experiments.experiments_utils import (
    setup_logger,
    create_base_parser,
    add_data_dir_arg,
    add_cache_dir_arg,
    add_train_sampling_arg,
    add_test_sampling_args,
    add_seed_arg,
    add_execution_args,
    build_default_cohorts,
    validate_files_exist,
)
from src.data_utils import load_tracked_patient_cohort
from src.gram import compute_asymmetric_normalized_kernel, ensure_mkl_weights
from src.evaluation import evaluate_patient_level_novelty
from src.model_io import load_svm_model

logger = setup_logger(__name__)

def main():
    parser = create_base_parser("Run Full Cohort Inference to replicate exact MIL experiment sequences")
    add_data_dir_arg(parser, required=True)
    add_cache_dir_arg(parser, project_root)
    add_train_sampling_arg(parser)
    add_test_sampling_args(parser)
    add_seed_arg(parser)
    add_execution_args(parser)
    parser.add_argument("--model-name", type=str, default="ocsvm_pretrained.pkl", help="Name of the saved model file")
    args = parser.parse_args()

    logger.info("=====================================================")
    logger.info(" EXACT INFERENCE REPLICATION USING SAVED MODEL")
    logger.info("=====================================================")

    train_normal_files, test_normal_files, test_tumor_files = build_default_cohorts(args.data_dir)
    all_files = train_normal_files + test_normal_files + test_tumor_files
    if not validate_files_exist(all_files, logger):
        sys.exit(1)

    # By passing all files and the same seed, we guarantee the exact same sequence of RNG states
    # as the all-in-one script.
    logger.info("\nStarting data loading and tracking...")
    _, test_data, _, test_files_info = load_tracked_patient_cohort(
        train_normal_files, 
        test_normal_files, 
        test_tumor_files,
        args.max_train,
        args.max_test_normal,
        args.max_test_tumor,
        args.seed,
        args.cache_dir,
    )

    model_path = os.path.join(project_root, "models", args.model_name)
    logger.info(f"\nLoading saved SVM model from {model_path}...")
    artifact = load_svm_model(model_path)
    mkl_weights = ensure_mkl_weights(artifact.max_k, artifact.mismatches, artifact.mkl_weights)

    start_time = time.time()
    
    logger.info("\nComputing Asymmetric Kernel for Testing...")
    K_test = compute_asymmetric_normalized_kernel(
        test_seqs=test_data,
        train_states=artifact.train_states,
        max_k=artifact.max_k,
        mismatches=artifact.mismatches,
        mkl_weights=mkl_weights,
        n_jobs=args.n_jobs
    )

    logger.info("\nPredicting sequence anomalies...")
    anomaly_scores = artifact.model.decision_function(K_test)
    
    patient_auc, per_patient_data = evaluate_patient_level_novelty(
        anomaly_scores, 
        test_files_info, 
    )
    
    # --- Optional: Generate Score Distribution Plots ---
    if args.plot_dir:
        from src.plotting import generate_score_distribution_plots
        final_plot_dir = os.path.join(args.plot_dir, f"m_{artifact.mismatches}", f"k_{artifact.max_k}")
        generate_score_distribution_plots(per_patient_data, final_plot_dir, args.seed)
    
    elapsed = time.time() - start_time
    logger.info(f"\nTotal Inference Execution Time: {elapsed:.2f} seconds")
    logger.info(f"PATIENT-LEVEL ROC-AUC: {patient_auc:.4f}")

if __name__ == "__main__":
    main()
