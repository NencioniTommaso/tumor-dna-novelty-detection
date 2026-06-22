#!/usr/bin/env python3
"""
run_nystrom_m_selection.py
Nyström m-Selection Experiment for Mismatch String Kernel.

Systematically evaluates Nyström approximation quality at various m/N ratios
to find the optimal number of landmark points for scaling to 100k+ sequences.

Uses the existing 30k-sequence pipeline as ground truth:
  1. Computes the exact Gram matrix (30k×30k) once
  2. Trains an exact OC-SVM baseline, evaluates on 10k test (5k healthy + 5k tumor)
  3. Sweeps m values from 1% to 10% of N, with 3 landmark seeds each
  4. For each m, measures:
     - Intrinsic: relative Frobenius error, spectral error, diagonal error
     - Downstream: AUC, Spearman rank correlation, score discrimination gap
  5. Outputs CSV results, multi-panel summary plot, and recommendation file

Usage
-----
    python experiments/run_nystrom_m_selection.py \\
        --data-dir /path/to/fasta_files \\
        --max-k 6 --mismatches 1 --nu-param 0.2 --n-jobs -1
"""

import csv
import gc
import os
import sys
import time

# ── Path setup ─────────────────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import numpy as np
import scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score

from src.data_utils import (
    load_training_cohort_tracked_indices,
    load_test_cohort_only,
)
from src.gram import (
    generate_mkl_weights,
    mixed_string_kernel,
    normalize_gram,
    compute_asymmetric_normalized_kernel,
)
from src.nystrom import (
    build_combined_test_features,
    normalize_rows,
    nystrom_fit,
    nystrom_transform,
)
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = setup_logger(__name__)

# ── Experiment constants ───────────────────────────────────────────────
M_RATIOS = [0.01, 0.02, 0.05, 0.10]
LANDMARK_SEEDS = [42, 123, 7]
INTRINSIC_SAMPLE_SIZE = 2000
DEFAULT_MAX_TRAIN = 30_000
DEFAULT_MAX_TEST_HEALTHY = 5_000
DEFAULT_MAX_TEST_TUMOR = 5_000


# ─────────────────────────────────────────────────────────────────────
# Intrinsic kernel approximation metrics
# ─────────────────────────────────────────────────────────────────────

def compute_intrinsic_metrics(
    K_norm_sample: np.ndarray,
    Phi_sample: np.ndarray,
) -> dict:
    """Compare Nyström-approximated kernel to exact kernel on a sampled submatrix.

    Parameters
    ----------
    K_norm_sample : np.ndarray, shape (s, s)
        Exact normalized kernel restricted to sampled row/column indices.
    Phi_sample : np.ndarray, shape (s, m)
        Nyström feature vectors for the same sampled indices.

    Returns
    -------
    dict
        Keys: rel_frobenius_error, rel_spectral_error, max_diag_error, mean_diag_error.
    """
    K_approx = Phi_sample @ Phi_sample.T
    diff = K_norm_sample - K_approx

    frob_exact = np.linalg.norm(K_norm_sample, 'fro')
    frob_err = np.linalg.norm(diff, 'fro')

    spectral_exact = np.linalg.norm(K_norm_sample, 2)
    spectral_err = np.linalg.norm(diff, 2)

    diag_diff = np.abs(np.diag(K_norm_sample) - np.diag(K_approx))

    return {
        'rel_frobenius_error': frob_err / max(frob_exact, 1e-12),
        'rel_spectral_error': spectral_err / max(spectral_exact, 1e-12),
        'max_diag_error': float(np.max(diag_diff)),
        'mean_diag_error': float(np.mean(diag_diff)),
    }


# ─────────────────────────────────────────────────────────────────────
# Downstream task metrics
# ─────────────────────────────────────────────────────────────────────

def compute_downstream_metrics(
    Phi_train: np.ndarray,
    Phi_test: np.ndarray,
    y_test: np.ndarray,
    exact_inv_scores: np.ndarray,
    nu: float,
) -> dict:
    """Train linear OC-SVM on Nyström features, compare with exact baseline.

    Parameters
    ----------
    Phi_train : np.ndarray, shape (N, m)
        Nyström training features.
    Phi_test : np.ndarray, shape (N_test, m)
        Nyström test features.
    y_test : np.ndarray, shape (N_test,)
        True labels (+1 healthy, -1 tumor).
    exact_inv_scores : np.ndarray, shape (N_test,)
        Inverted exact-kernel anomaly scores (higher = more anomalous).
    nu : float
        OC-SVM nu parameter.

    Returns
    -------
    dict
        Keys: auc, spearman_corr, spearman_pval, score_gap,
              mean_healthy_score, mean_tumor_score.
    """
    svm = OneClassSVM(kernel='linear', nu=nu)
    svm.fit(Phi_train)

    nystrom_raw = svm.decision_function(Phi_test)
    nystrom_inv = -nystrom_raw  # higher = more anomalous

    auc = roc_auc_score(y_test == -1, nystrom_inv)
    corr, pval = spearmanr(nystrom_inv, exact_inv_scores)

    h_mask = y_test == 1
    t_mask = y_test == -1

    return {
        'auc': auc,
        'spearman_corr': corr,
        'spearman_pval': pval,
        'score_gap': float(np.mean(nystrom_inv[t_mask]) - np.mean(nystrom_inv[h_mask])),
        'mean_healthy_score': float(np.mean(nystrom_inv[h_mask])),
        'mean_tumor_score': float(np.mean(nystrom_inv[t_mask])),
    }


# ─────────────────────────────────────────────────────────────────────
# Plot generation
# ─────────────────────────────────────────────────────────────────────

def generate_summary_plot(
    results: list[dict],
    exact_auc: float,
    exact_gap: float,
    N: int,
    out_path: str,
) -> None:
    """Generate a 2×2 multi-panel summary plot.

    Panels: (1) Frobenius error, (2) AUC, (3) Spearman correlation,
    (4) Score discrimination gap.  Each with mean ± std across seeds.
    """
    ratios = sorted(set(r['m_ratio'] for r in results))

    def gather(key):
        means, stds = [], []
        for ratio in ratios:
            vals = [r[key] for r in results if r['m_ratio'] == ratio]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        return np.array(means), np.array(stds)

    frob_mean, frob_std = gather('rel_frobenius_error')
    auc_mean, auc_std = gather('auc')
    corr_mean, corr_std = gather('spearman_corr')
    gap_mean, gap_std = gather('score_gap')

    m_values = [max(1, int(r * N)) for r in ratios]
    x_labels = [f"{r:.0%}\n(m={m:,})" for r, m in zip(ratios, m_values)]
    x = np.arange(len(ratios))

    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'figure.facecolor': 'white',
    })

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f'Nyström m-Selection Experiment  (N = {N:,})',
        fontsize=14, fontweight='bold', y=0.98,
    )

    color = '#2563eb'
    baseline_color = '#dc2626'

    # ── Panel 1: Frobenius error ──
    ax = axes[0, 0]
    ax.errorbar(x, frob_mean, yerr=frob_std, fmt='o-', color=color,
                capsize=4, markersize=6, linewidth=1.5)
    ax.set_ylabel('Relative Frobenius Error')
    ax.set_title('Kernel Approximation Error')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_xlabel('m / N  (m value)')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.10, color='gray', linestyle=':', alpha=0.5, label='10% threshold')
    ax.legend(fontsize=8)

    # ── Panel 2: AUC ──
    ax = axes[0, 1]
    ax.errorbar(x, auc_mean, yerr=auc_std, fmt='o-', color=color,
                capsize=4, markersize=6, linewidth=1.5, label='Nyström')
    ax.axhline(y=exact_auc, color=baseline_color, linestyle='--',
               linewidth=1.5, label=f'Exact (AUC = {exact_auc:.4f})')
    ax.set_ylabel('ROC-AUC')
    ax.set_title('Downstream Task Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_xlabel('m / N  (m value)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # ── Panel 3: Spearman correlation ──
    ax = axes[1, 0]
    ax.errorbar(x, corr_mean, yerr=corr_std, fmt='o-', color=color,
                capsize=4, markersize=6, linewidth=1.5)
    ax.set_ylabel('Spearman Rank Correlation')
    ax.set_title('Score Ranking Preservation')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_xlabel('m / N  (m value)')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.95, color='gray', linestyle=':', alpha=0.5, label='ρ = 0.95')
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.2)
    ax.legend(fontsize=8)

    # ── Panel 4: Score gap ──
    ax = axes[1, 1]
    ax.errorbar(x, gap_mean, yerr=gap_std, fmt='o-', color=color,
                capsize=4, markersize=6, linewidth=1.5, label='Nyström')
    ax.axhline(y=exact_gap, color=baseline_color, linestyle='--',
               linewidth=1.5, label=f'Exact (gap = {exact_gap:.4f})')
    ax.set_ylabel('Score Gap  (Tumor − Healthy)')
    ax.set_title('Discrimination Power')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_xlabel('m / N  (m value)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save both PDF and PNG
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    base, ext = os.path.splitext(out_path)
    alt_ext = '.png' if ext == '.pdf' else '.pdf'
    plt.savefig(base + alt_ext, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Summary plot saved to {out_path} (+ {alt_ext})")


# ─────────────────────────────────────────────────────────────────────
# Recommendation generator
# ─────────────────────────────────────────────────────────────────────

def write_recommendation(
    results: list[dict],
    exact_auc: float,
    exact_gap: float,
    N: int,
    out_path: str,
) -> str:
    """Analyze the sweep results and write a recommendation file.

    Selection criteria (smallest m that satisfies both):
      - AUC within 0.02 of exact
      - Spearman rank correlation > 0.95
    """
    ratios = sorted(set(r['m_ratio'] for r in results))

    lines = [
        "=" * 70,
        "NYSTRÖM m-SELECTION EXPERIMENT — RECOMMENDATION",
        "=" * 70,
        "",
        f"Training set size N = {N:,}",
        f"Exact baseline:  AUC = {exact_auc:.4f}    Score gap = {exact_gap:.4f}",
        "",
        f"{'m/N':>8s} {'m':>7s} {'Frob':>9s} {'Spectral':>9s} {'AUC':>9s} "
        f"{'Corr':>9s} {'Gap':>9s} {'Time(s)':>8s}",
        "-" * 70,
    ]

    best_ratio = None

    for ratio in ratios:
        subset = [r for r in results if r['m_ratio'] == ratio]
        m_val = subset[0]['m']
        frob = np.mean([r['rel_frobenius_error'] for r in subset])
        spec = np.mean([r['rel_spectral_error'] for r in subset])
        auc = np.mean([r['auc'] for r in subset])
        corr = np.mean([r['spearman_corr'] for r in subset])
        gap = np.mean([r['score_gap'] for r in subset])
        elapsed = np.mean([r['elapsed_s'] for r in subset])

        marker = ""
        auc_ok = abs(auc - exact_auc) < 0.02
        corr_ok = corr > 0.95
        if auc_ok and corr_ok and best_ratio is None:
            best_ratio = ratio
            marker = "  <── RECOMMENDED"

        lines.append(
            f"{ratio:>8.2%} {m_val:>7,d} {frob:>9.4f} {spec:>9.4f} {auc:>9.4f} "
            f"{corr:>9.4f} {gap:>9.4f} {elapsed:>8.1f}{marker}"
        )

    lines.append("-" * 70)
    lines.append("")

    if best_ratio is not None:
        m_100k = int(best_ratio * 100_000)
        m_120k = int(best_ratio * 120_000)
        lines.extend([
            f"RECOMMENDED m/N ratio: {best_ratio:.2%}",
            f"  At N = 100,000  →  m = {m_100k:,}",
            f"  At N = 120,000  →  m = {m_120k:,}",
            "",
            "Justification:",
            f"  Smallest ratio where AUC is within 0.02 of exact ({exact_auc:.4f})",
            f"  AND Spearman rank correlation > 0.95.",
        ])
    else:
        lines.extend([
            "WARNING: No ratio met both criteria (AUC within 0.02 + Corr > 0.95).",
            "Consider using the largest tested ratio, or investigate further.",
        ])

    lines.extend(["", "=" * 70])

    text = "\n".join(lines)
    with open(out_path, 'w') as f:
        f.write(text + "\n")
    logger.info(f"Recommendation saved to {out_path}")
    return text


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = create_base_parser(
        "Nyström m-Selection Experiment for Mismatch String Kernel"
    )
    add_data_dir_arg(parser, required=True)
    add_cache_dir_arg(parser, project_root)
    add_kernel_args(parser)
    add_nu_arg(parser)
    add_execution_args(parser)
    parser.add_argument(
        "--max-train", type=int, default=DEFAULT_MAX_TRAIN,
        help=f"Total training sequences (default: {DEFAULT_MAX_TRAIN}).",
    )
    parser.add_argument(
        "--max-test-healthy", type=int, default=DEFAULT_MAX_TEST_HEALTHY,
        help=f"Healthy test sequences (default: {DEFAULT_MAX_TEST_HEALTHY}).",
    )
    parser.add_argument(
        "--max-test-tumor", type=int, default=DEFAULT_MAX_TEST_TUMOR,
        help=f"Tumor test sequences (default: {DEFAULT_MAX_TEST_TUMOR}).",
    )
    parser.add_argument(
        "--held-out-id", type=int, default=7,
        help="ID of the healthy patient to hold out (2–7, default: 7).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for data loading (default: 42).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: results/nystrom_m_selection/m_<m>/k_<k>/).",
    )
    args = parser.parse_args()

    # ── Build fold ────────────────────────────────────────────────────
    fold = build_loo_single_fold(args.data_dir, args.held_out_id)
    fold_name = fold['fold_name']

    all_files = fold['train_files'] + [fold['held_out_file']] + fold['tumor_files']
    if not validate_files_exist(all_files, logger):
        sys.exit(1)

    # ── Output directory ──────────────────────────────────────────────
    out_dir = args.output_dir or os.path.join(
        project_root, "results", "nystrom_m_selection",
        f"m_{args.mismatches}", f"k_{args.max_k}",
    )
    os.makedirs(out_dir, exist_ok=True)

    # ── MKL weights ───────────────────────────────────────────────────
    mkl_weights = generate_mkl_weights(
        args.max_k, noise_threshold=max(1, 2 * args.mismatches)
    )

    # ── Banner ────────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info(" NYSTRÖM m-SELECTION EXPERIMENT")
    logger.info("=" * 70)
    logger.info(f"  Fold              : {fold_name}")
    logger.info(f"  Training patients : {[os.path.basename(f) for f in fold['train_files']]}")
    logger.info(f"  Held-out healthy  : {os.path.basename(fold['held_out_file'])}")
    logger.info(f"  Tumor patients    : {[os.path.basename(f) for f in fold['tumor_files']]}")
    logger.info(f"  Max train seqs    : {args.max_train}")
    logger.info(f"  Test healthy/tumor: {args.max_test_healthy} / {args.max_test_tumor}")
    logger.info(f"  Kernel            : max_k={args.max_k}, m={args.mismatches}")
    logger.info(f"  MKL weights       : {mkl_weights}")
    logger.info(f"  OCSVM nu          : {args.nu_param}")
    logger.info(f"  m/N ratios        : {M_RATIOS}")
    logger.info(f"  Landmark seeds    : {LANDMARK_SEEDS}")
    logger.info(f"  Seed              : {args.seed}")
    logger.info(f"  Output dir        : {out_dir}")
    logger.info("=" * 70)

    total_start = time.time()

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1: Load Data
    # ══════════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("─── Phase 1: Loading Data ───")

    train_data, _ = load_training_cohort_tracked_indices(
        fold['train_files'], args.max_train, args.seed, args.cache_dir,
    )
    N = len(train_data)
    logger.info(f"Training: {N:,} sequences")

    # Use only the first tumor patient (Colo_11) with the full 5k budget
    test_data, y_test, _ = load_test_cohort_only(
        [fold['held_out_file']], [fold['tumor_files'][0]],
        args.max_test_healthy, args.max_test_tumor,
        args.seed, args.cache_dir,
    )
    N_test = len(test_data)
    n_healthy_test = int(np.sum(y_test == 1))
    n_tumor_test = int(np.sum(y_test == -1))
    logger.info(f"Test: {N_test:,} sequences ({n_healthy_test:,} healthy, {n_tumor_test:,} tumor)")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2: Exact Gram Matrix (Ground Truth)
    # ══════════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("─── Phase 2: Computing Exact Gram Matrix ───")
    t0 = time.time()

    K_train_unnorm, train_states = mixed_string_kernel(
        sequences=train_data, k_max=args.max_k, m=args.mismatches,
        weights=mkl_weights, n_jobs=args.n_jobs,
    )
    K_norm = normalize_gram(K_train_unnorm)
    del K_train_unnorm
    gc.collect()

    gram_time = time.time() - t0
    logger.info(f"Exact Gram matrix ({N:,} × {N:,}) computed in {gram_time:.1f}s")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 3: Exact OC-SVM Baseline
    # ══════════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("─── Phase 3: Exact OC-SVM Baseline ───")

    svm_exact = OneClassSVM(kernel='precomputed', nu=args.nu_param)
    svm_exact.fit(K_norm)
    logger.info(f"Exact OC-SVM fitted (nu={args.nu_param})")

    logger.info("Computing exact asymmetric test kernel...")
    t0 = time.time()
    K_test = compute_asymmetric_normalized_kernel(
        test_seqs=test_data, train_states=train_states,
        max_k=args.max_k, mismatches=args.mismatches,
        mkl_weights=mkl_weights, n_jobs=args.n_jobs,
    )
    logger.info(f"Asymmetric kernel ({K_test.shape[0]:,} × {K_test.shape[1]:,}) "
                f"computed in {time.time() - t0:.1f}s")

    exact_raw_scores = svm_exact.decision_function(K_test)
    exact_inv_scores = -exact_raw_scores  # higher = more anomalous
    exact_auc = roc_auc_score(y_test == -1, exact_inv_scores)
    exact_healthy_mean = float(np.mean(exact_inv_scores[y_test == 1]))
    exact_tumor_mean = float(np.mean(exact_inv_scores[y_test == -1]))
    exact_gap = exact_tumor_mean - exact_healthy_mean

    logger.info(f"Exact baseline:  AUC = {exact_auc:.4f}")
    logger.info(f"  Healthy mean score = {exact_healthy_mean:.4f}")
    logger.info(f"  Tumor mean score   = {exact_tumor_mean:.4f}")
    logger.info(f"  Score gap          = {exact_gap:.4f}")

    del K_test, svm_exact
    gc.collect()

    # ══════════════════════════════════════════════════════════════════
    # PHASE 4: Build Combined Feature Matrix for Nyström
    #          (reuses per-k features from Phase 2 — no re-extraction)
    # ══════════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("─── Phase 4: Building Nyström Feature Matrices ───")

    # Reuse the per-k weighted sparse features already computed in mixed_string_kernel
    active_ks = sorted(train_states.keys())
    blocks = [train_states[k]['X_train'] for k in active_ks]
    per_k_vocabs = {k: train_states[k]['vocabulary'] for k in active_ks}
    X_combined = sp.hstack(blocks, format='csr')

    logger.info(
        f"Training feature matrix: {X_combined.shape[0]:,} × {X_combined.shape[1]:,} "
        f"(nnz={X_combined.nnz:,}, reused from gram computation)"
    )

    del train_states, blocks
    gc.collect()

    X_norm_train, _ = normalize_rows(X_combined)
    del X_combined
    gc.collect()
    logger.info("Training features row-normalized")

    # Build test features in the same feature space (using training vocabularies)
    logger.info("Extracting test features...")
    t0 = time.time()
    X_test_combined = build_combined_test_features(
        test_data, per_k_vocabs,
        args.max_k, args.mismatches, mkl_weights,
        n_jobs=args.n_jobs,
    )
    X_norm_test, _ = normalize_rows(X_test_combined)
    del X_test_combined
    gc.collect()
    logger.info(f"Test features ({X_norm_test.shape[0]:,} × {X_norm_test.shape[1]:,}) "
                f"extracted and normalized in {time.time() - t0:.1f}s")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 5: Prepare Intrinsic Evaluation Sample
    # ══════════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("─── Phase 5: Preparing Intrinsic Evaluation ───")

    rng = np.random.RandomState(args.seed)
    sample_size = min(INTRINSIC_SAMPLE_SIZE, N)
    sample_idx = rng.choice(N, sample_size, replace=False)
    sample_idx.sort()

    K_norm_sample = K_norm[np.ix_(sample_idx, sample_idx)].copy()
    logger.info(f"Sampled {sample_size:,} × {sample_size:,} submatrix for intrinsic evaluation")

    # Free the full Gram matrix — this is the largest memory save (~7 GB for 30k)
    del K_norm
    gc.collect()
    logger.info("Released exact Gram matrix from memory")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 6: Nyström m-Sweep
    # ══════════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=" * 70)
    logger.info(" PHASE 6: NYSTRÖM m-SWEEP")
    logger.info("=" * 70)

    all_results: list[dict] = []

    for ratio in M_RATIOS:
        m = max(1, int(ratio * N))
        logger.info("")
        logger.info(f"━━━ m/N = {ratio:.0%}  (m = {m:,}, N = {N:,}) ━━━")

        for lm_seed in LANDMARK_SEEDS:
            t_iter = time.time()

            # ── Fit ──
            state = nystrom_fit(
                X_norm_train, m, lm_seed,
                per_k_vocabs, mkl_weights,
                args.max_k, args.mismatches,
            )

            # ── Transform training data ──
            Phi_train = nystrom_transform(X_norm_train, state, n_jobs=args.n_jobs)

            # ── Transform test data ──
            Phi_test = nystrom_transform(X_norm_test, state, n_jobs=args.n_jobs)

            # ── Intrinsic metrics (on sampled submatrix) ──
            Phi_sample = Phi_train[sample_idx]
            intrinsic = compute_intrinsic_metrics(K_norm_sample, Phi_sample)
            del Phi_sample

            # ── Downstream metrics ──
            downstream = compute_downstream_metrics(
                Phi_train, Phi_test, y_test, exact_inv_scores, args.nu_param,
            )

            elapsed = time.time() - t_iter

            # ── Record ──
            row = {
                'm_ratio': ratio,
                'm': m,
                'landmark_seed': lm_seed,
                'n_components_effective': state.n_components,
                'elapsed_s': round(elapsed, 2),
                **intrinsic,
                **downstream,
            }
            all_results.append(row)

            logger.info(
                f"  seed={lm_seed:>3d} │ Frob={intrinsic['rel_frobenius_error']:.4f} │ "
                f"AUC={downstream['auc']:.4f} │ Corr={downstream['spearman_corr']:.4f} │ "
                f"Gap={downstream['score_gap']:.4f} │ {elapsed:.1f}s"
            )

            del Phi_train, Phi_test, state
            gc.collect()

    # ══════════════════════════════════════════════════════════════════
    # PHASE 7: Save Results
    # ══════════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("─── Phase 7: Saving Results ───")

    # ── CSV ──
    csv_path = os.path.join(out_dir, "m_selection_metrics.csv")
    fieldnames = list(all_results[0].keys())
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            writer.writerow({
                k: (f"{v:.6f}" if isinstance(v, float) else v)
                for k, v in row.items()
            })
    logger.info(f"Metrics CSV saved to {csv_path}")

    # ── Summary plot ──
    plot_path = os.path.join(out_dir, "m_selection_summary.pdf")
    generate_summary_plot(all_results, exact_auc, exact_gap, N, plot_path)

    # ── Recommendation ──
    rec_path = os.path.join(out_dir, "m_selection_recommendation.txt")
    rec_text = write_recommendation(all_results, exact_auc, exact_gap, N, rec_path)

    # ── Exact baseline reference file ──
    baseline_path = os.path.join(out_dir, "exact_baseline.txt")
    with open(baseline_path, 'w') as f:
        f.write(f"Exact OC-SVM Baseline (N = {N:,})\n")
        f.write(f"AUC              : {exact_auc:.6f}\n")
        f.write(f"Healthy mean     : {exact_healthy_mean:.6f}\n")
        f.write(f"Tumor mean       : {exact_tumor_mean:.6f}\n")
        f.write(f"Score gap        : {exact_gap:.6f}\n")
        f.write(f"Gram time (s)    : {gram_time:.1f}\n")
        f.write(f"Kernel           : max_k={args.max_k}, m={args.mismatches}\n")
        f.write(f"nu               : {args.nu_param}\n")
        f.write(f"Test set         : {n_healthy_test} healthy + {n_tumor_test} tumor\n")
    logger.info(f"Baseline reference saved to {baseline_path}")

    # ══════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════
    total_elapsed = time.time() - total_start

    logger.info("")
    logger.info("=" * 70)
    logger.info(" EXPERIMENT COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Exact baseline AUC  : {exact_auc:.4f}")
    logger.info(f"  Exact score gap     : {exact_gap:.4f}")
    logger.info("")

    ratios = sorted(set(r['m_ratio'] for r in all_results))
    logger.info(f"  {'m/N':>8s} {'m':>7s} {'Frob':>8s} {'Spec':>8s} "
                f"{'AUC':>8s} {'Corr':>8s} {'Gap':>8s}")
    logger.info("  " + "-" * 60)
    for ratio in ratios:
        subset = [r for r in all_results if r['m_ratio'] == ratio]
        m_val = subset[0]['m']
        frob = np.mean([r['rel_frobenius_error'] for r in subset])
        spec = np.mean([r['rel_spectral_error'] for r in subset])
        auc = np.mean([r['auc'] for r in subset])
        corr = np.mean([r['spearman_corr'] for r in subset])
        gap = np.mean([r['score_gap'] for r in subset])
        logger.info(
            f"  {ratio:>8.2%} {m_val:>7,d} {frob:>8.4f} {spec:>8.4f} "
            f"{auc:>8.4f} {corr:>8.4f} {gap:>8.4f}"
        )

    logger.info("")
    logger.info(f"  Total time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    logger.info(f"  Results   : {out_dir}")
    logger.info("=" * 70)

    # Print recommendation to stdout
    logger.info("")
    logger.info(rec_text)


if __name__ == '__main__':
    main()
