"""
plot_aggregation_heatmap_folds.py
Produces a heatmap of AUC (normalized Mann-Whitney U statistic) by score
aggregation method × number of reads sub-sampled.

Mean and standard deviation are calculated across the LOO folds (since there are no rounds).
Supports both LOO-only (default) and all-healthy (--all-healthy) evaluations.
"""

import argparse
import glob
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── Path setup ────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

# ── Constants ─────────────────────────────────────────────────────────
N_VALUES = [1_000, 5_000, 10_000, 20_000, 50_000]

AGGREGATORS = {
    "max":              lambda s: np.max(s),
    "mean":             lambda s: np.mean(s),
    "median":           lambda s: np.median(s),
    "percentile_90":    lambda s: np.percentile(s, 90),
    "percentile_95":    lambda s: np.percentile(s, 95),
    "percentile_99":    lambda s: np.percentile(s, 99),
    "topk_mean_100":    lambda s: np.mean(np.sort(s)[-100:]),
    "topk_mean_500":    lambda s: np.mean(np.sort(s)[-500:]),
}

AGG_ORDER = [
    "max", "mean", "median",
    "percentile_90", "percentile_95", "percentile_99",
    "topk_mean_100", "topk_mean_500",
]

SUBSAMPLE_SEED = 42

def load_subject_scores(csv_path: str) -> np.ndarray:
    scores = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if scores.ndim == 2 and scores.shape[1] == 1:
        scores = scores.squeeze()
    return scores

def classify_subject(filename: str) -> str:
    basename = os.path.splitext(os.path.basename(filename))[0]
    if basename.lower().startswith("colo"):
        return "tumor"
    elif basename.lower().startswith("healthy"):
        return "healthy"
    return "unknown"

def subject_name_from_file(csv_path: str) -> str:
    basename = os.path.basename(csv_path)
    return basename.split("_scores_")[0]

def loo_subject_from_dir(loo_dir: str) -> str:
    dirname = os.path.basename(loo_dir)
    return dirname.replace("LOO_", "")

def auc_healthy_vs_sick(healthy_scores: np.ndarray, sick_scores: np.ndarray) -> float:
    h = np.asarray(healthy_scores, dtype=np.float64)
    s = np.asarray(sick_scores, dtype=np.float64)
    n_h, n_s = len(h), len(s)
    if n_h == 0 or n_s == 0:
        return float("nan")

    combined = np.concatenate([h, s])
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(combined) + 1)

    _, inv, counts = np.unique(combined, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        avg_rank = np.zeros_like(counts, dtype=np.float64)
        cum = 0
        for i, c in enumerate(counts):
            avg_rank[i] = cum + (c + 1) / 2
            cum += c
        ranks = avg_rank[inv]

    sum_ranks_sick = ranks[n_h:].sum()
    u_sick = sum_ranks_sick - n_s * (n_s + 1) / 2
    return float(u_sick / (n_h * n_s))

def compute_heatmap_data(loo_dirs: list[str], all_healthy: bool) -> tuple[np.ndarray, np.ndarray]:
    n_aggs = len(AGG_ORDER)
    n_ns = len(N_VALUES)

    all_aucs = {
        (a, n): [] for a in AGG_ORDER for n in N_VALUES
    }

    rng = np.random.default_rng(SUBSAMPLE_SEED)

    for loo_dir in loo_dirs:
        loo_subject = loo_subject_from_dir(loo_dir)
        print(f"\n{'─' * 60}")
        print(f"  Processing fold: {os.path.basename(loo_dir)}  (LOO subject: {loo_subject})")
        print(f"{'─' * 60}")

        csv_files = sorted(glob.glob(os.path.join(loo_dir, "*_scores_*.csv")))
        if not csv_files:
            continue

        healthy_files = []
        tumor_files = []

        for f in csv_files:
            subj = subject_name_from_file(f)
            kind = classify_subject(f)
            if kind == "healthy":
                if all_healthy or subj == loo_subject:
                    healthy_files.append(f)
            elif kind == "tumor":
                tumor_files.append(f)

        if not healthy_files or not tumor_files:
            print(f"  WARNING: Missing healthy or tumor files in {loo_dir}, skipping.")
            continue

        healthy_scores_all = []
        for hf in healthy_files:
            healthy_scores_all.append(load_subject_scores(hf))

        tumor_scores_all = []
        for tf in tumor_files:
            tumor_scores_all.append(load_subject_scores(tf))

        print(f"  Healthy test subjects: {len(healthy_files)}")
        print(f"  Tumor test subjects: {len(tumor_files)}")

        for n_reads in N_VALUES:
            h_subs = []
            for h_scores in healthy_scores_all:
                n_seqs = len(h_scores)
                if n_reads >= n_seqs:
                    h_subs.append(h_scores)
                else:
                    idx_h = rng.choice(n_seqs, size=n_reads, replace=False)
                    h_subs.append(h_scores[idx_h])
            
            t_subs = []
            for t_scores in tumor_scores_all:
                n_seqs = len(t_scores)
                if n_reads >= n_seqs:
                    t_subs.append(t_scores)
                else:
                    idx_t = rng.choice(n_seqs, size=n_reads, replace=False)
                    t_subs.append(t_scores[idx_t])

            for agg_name in AGG_ORDER:
                agg_fn = AGGREGATORS[agg_name]
                h_aggs = np.array([agg_fn(hs) for hs in h_subs])
                t_aggs = np.array([agg_fn(ts) for ts in t_subs])
                
                auc = auc_healthy_vs_sick(h_aggs, t_aggs)
                all_aucs[(agg_name, n_reads)].append(auc)

    mean_matrix = np.zeros((n_aggs, n_ns))
    std_matrix = np.zeros((n_aggs, n_ns))

    for i, agg_name in enumerate(AGG_ORDER):
        for j, n_reads in enumerate(N_VALUES):
            aucs = np.array(all_aucs[(agg_name, n_reads)])
            if len(aucs) > 0:
                mean_matrix[i, j] = np.mean(aucs)
                std_matrix[i, j] = np.std(aucs)
            else:
                mean_matrix[i, j] = np.nan
                std_matrix[i, j] = np.nan

    return mean_matrix, std_matrix

def plot_heatmap(mean_matrix, std_matrix, output_path, title):
    n_aggs, n_ns = mean_matrix.shape
    annot = np.empty((n_aggs, n_ns), dtype=object)
    for i in range(n_aggs):
        for j in range(n_ns):
            m = mean_matrix[i, j]
            s = std_matrix[i, j]
            if np.isnan(m):
                annot[i, j] = "N/A"
            else:
                annot[i, j] = f"{m:.3f}\n±{s:.3f}"

    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = sns.color_palette("vlag", as_cmap=True)

    valid = mean_matrix[~np.isnan(mean_matrix)]
    vmin = max(0.0, valid.min() - 0.05) if len(valid) > 0 else 0.0
    vmax = min(1.0, valid.max() + 0.05) if len(valid) > 0 else 1.0

    sns.heatmap(
        mean_matrix,
        annot=annot,
        fmt="",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        xticklabels=[f"{n:,}" for n in N_VALUES],
        yticklabels=AGG_ORDER,
        cbar_kws={"label": "Mean AUC", "shrink": 0.8},
        ax=ax,
        annot_kws={"fontsize": 11, "fontweight": "bold"},
    )

    ax.set_xlabel("Reads per Patient", fontsize=13, labelpad=10)
    ax.set_ylabel("Score Aggregation Method", fontsize=13, labelpad=10)
    ax.set_title(title, fontsize=14, pad=15)

    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.5)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=11)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nHeatmap saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Heatmap of AUC by aggregation method × N reads across LOO folds.")
    parser.add_argument("--results-dir", required=True, help="Path containing the LOO_* fold directories.")
    parser.add_argument("--output", default=None, help="Output path for the heatmap PNG.")
    parser.add_argument("--title", default=None, help="Custom plot title.")
    parser.add_argument("--all-healthy", action="store_true", help="Include all healthy patients in AUC calculation, not just the LOO subject.")
    args = parser.parse_args()

    loo_dirs = sorted(glob.glob(os.path.join(args.results_dir, "LOO_*")))
    if not loo_dirs:
        print(f"ERROR: No LOO_* directories found in {args.results_dir}")
        sys.exit(1)

    print(f"Found {len(loo_dirs)} LOO fold(s): {[os.path.basename(d) for d in loo_dirs]}")

    mean_matrix, std_matrix = compute_heatmap_data(loo_dirs, args.all_healthy)
    
    if args.output is None:
        suffix = "_all_healthy" if args.all_healthy else ""
        output_path = os.path.join(args.results_dir, f"heatmap_aggregation_auc_folds{suffix}.png")
    else:
        output_path = args.output
        
    if args.title is None:
        mode_str = "All Healthy" if args.all_healthy else "LOO-Only Healthy"
        title = f"AUC by Aggregation × N Reads ({mode_str})\nMean ± Std across 7 Folds"
    else:
        title = args.title

    plot_heatmap(mean_matrix, std_matrix, output_path, title=title)

if __name__ == "__main__":
    main()
