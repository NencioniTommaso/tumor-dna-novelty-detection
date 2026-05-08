"""
diagnose_diagonal.py
A diagnostic tool to visualize Gram Matrix diagonal dominance.
Helps tune 'k' and 'm' (mismatches) before running full-scale experiments.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Dynamically resolve paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.data_utils import load_patient_cohort
from src.kernels import mixed_string_kernel, normalize_gram
from experiments.experiments_utils import (
    setup_logger,
    create_base_parser,
    add_data_dir_arg,
    add_cache_dir_arg,
)

logger = setup_logger(__name__)

def parse_diag_args():
    parser = create_base_parser("Diagnose Gram Matrix Diagonal Dominance")
    add_data_dir_arg(parser, required=True)
    add_cache_dir_arg(parser, project_root)
                        
    parser.add_argument("--k", type=int, default=6, help="Specific K-mer size to test.")
    parser.add_argument("--mismatches", type=int, default=0, help="Mismatches to allow.")
    parser.add_argument("--samples", type=int, default=500, help="Total sequences to sample.")
    
    # UPDATED: Default output path now routes to experiments/heatmaps/
    default_out_path = os.path.join(current_dir, "heatmaps", "gram_heatmap.png")
    parser.add_argument("--out", type=str, default=default_out_path, help="Output image filepath.")
    
    return parser.parse_args()

def main():
    args = parse_diag_args()
    logger.info(f"--- DIAGNOSTIC MODE: k={args.k}, m={args.mismatches} ---")
    
    # 1. Quick Data Load (Small Sample)
    train_normal = [os.path.join(args.data_dir, "Healthy_2_merged_subset_1200000.fa")]
    test_tumor = [os.path.join(args.data_dir, "Colo_11_merged_subset_1200000.fa")]
    
    # Split the requested samples evenly between Normal and Tumor
    half_samples = args.samples // 2
    
    train_data, test_data, labels = load_patient_cohort(
        train_normal_files=train_normal,
        test_normal_files=[],
        test_tumor_files=test_tumor,
        max_train=half_samples,
        max_test_normal=0,
        max_test_tumor=half_samples,
        random_seed=42,
        index_cache_dir=args.cache_dir
    )
    
    all_data = train_data + test_data
    actual_samples = len(all_data)
    logger.info(f"Loaded {actual_samples} sequences for visualization.")
    
    # 2. Compute the exact K-mer Gram Matrix
    weights = [0.0] * args.k
    weights[-1] = 1.0  
    
    logger.info("Computing kernel...")
    K, _ = mixed_string_kernel(
        sequences=all_data,
        k_max=args.k,
        m=args.mismatches,
        weights=weights,
        n_jobs=-1
    )
    
    logger.info("Normalizing...")
    K_norm = normalize_gram(K)
    
    # 3. Mathematical Diagnosis
    off_diagonals = K_norm[~np.eye(K_norm.shape[0], dtype=bool)]
    mean_similarity = np.mean(off_diagonals)
    max_off_diag = np.max(off_diagonals)
    
    logger.info("=========================================")
    logger.info(" MATRIX DIAGNOSTICS")
    logger.info("=========================================")
    logger.info(f"Mean Off-Diagonal Similarity : {mean_similarity:.6f}")
    logger.info(f"Max Off-Diagonal Similarity  : {max_off_diag:.6f}")
    
    if mean_similarity < 0.01:
        logger.warning("DANGER: Matrix is highly diagonal! The SVM will struggle to generalize.")
        logger.warning("Recommendation: Increase --mismatches or decrease --k.")
    elif mean_similarity > 0.5:
        logger.warning("DANGER: Matrix is too dense! Sequences look too similar.")
        logger.warning("Recommendation: Decrease --mismatches or increase --k.")
    else:
        logger.info("SUCCESS: Matrix shows a healthy balance of sparsity and similarity.")
        
    # 4. Visual Diagnosis
    # UPDATED: Ensure the output directory exists before saving
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    logger.info(f"Generating heatmap and saving to {args.out}...")
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(K_norm, cmap='viridis', vmin=0, vmax=1.0, 
                xticklabels=False, yticklabels=False)
    
    plt.title(f"Normalized Gram Matrix (N={actual_samples})\nk={args.k}, mismatches={args.mismatches} | Mean Off-Diag: {mean_similarity:.4f}")
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    logger.info("Done.")

if __name__ == "__main__":
    main()