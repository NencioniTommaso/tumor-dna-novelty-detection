"""
plot_aggregation_heatmap_pooled_all_healthy.py
Produces a heatmap of AUC (normalized Mann-Whitney U statistic) by score
aggregation method × number of reads sub-sampled.

This version pools ALL rounds together into one giant pool of fragments
per patient (e.g. 140,000 fragments if 20k reads x 7 rounds) and then 
uses repeated random subsampling (bootstrap) to compute mean and std.
It uses ALL healthy patients vs ALL tumor patients.

For each LOO fold, each bootstrap iteration, and each (aggregator, N):
  1. Sub-sample N reads from each patient's pooled scores.
  2. Aggregate the N scores into a single patient-level score.
  3. Compute AUC as the normalized Mann-Whitney U statistic
     (ALL healthy vs. ALL tumors).
  4. Average AUC ± std across iterations and LOO folds.

The heatmap has:
  - x-axis: N reads (1000, 5000, 10000, 20000, 50000, 100000, 140000)
  - y-axis: aggregation method
  - cell annotation: mean AUC ± std

Usage
-----
    python experiments/plot_aggregation_heatmap_pooled_all_healthy.py \
        --results-dir results/loo_multiround_nystrom_001/m_1/k_6
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
N_VALUES = [1_000, 5_000, 10_000, 20_000, 50_000, 100_000, 140_000]
N_BOOTSTRAP = 10

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


# ── Data loading ──────────────────────────────────────────────────────

def load_subject_scores(csv_path: str) -> np.ndarray:
    """Load a subject's per-round score CSV into a 2-D array."""
    # Using numpy for speed; skip the header row
    return np.loadtxt(csv_path, delimiter=",", skiprows=1)


def classify_subject(filename: str) -> str:
    """Return 'healthy' or 'tumor' from the filename."""
    basename = os.path.splitext(os.path.basename(filename))[0]
    if basename.lower().startswith("colo"):
        return "tumor"
    elif basename.lower().startswith("healthy"):
        return "healthy"
    return "unknown"


def subject_name_from_file(csv_path: str) -> str:
    """Extract subject name (e.g. 'Healthy_7') from filename."""
    basename = os.path.basename(csv_path)
    # Remove _scores_seed*.csv
    return basename.split("_scores_")[0]


def loo_subject_from_dir(loo_dir: str) -> str:
    """Extract the left-out healthy subject from the directory name."""
    dirname = os.path.basename(loo_dir)
    return dirname.replace("LOO_", "")


# ── AUC via Mann-Whitney U ───────────────────────────────────────────

def auc_healthy_vs_sick(
    healthy_scores: np.ndarray,
    sick_scores: np.ndarray,
) -> float:
    """AUC della classificazione sani-vs-malati usando lo score come discriminante."""
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


# ── Main computation ─────────────────────────────────────────────────

def compute_heatmap_data(loo_dirs: list[str]) -> tuple[np.ndarray, np.ndarray]:
    n_aggs = len(AGG_ORDER)
    n_ns = len(N_VALUES)

    all_aucs = {(a, n): [] for a in AGG_ORDER for n in N_VALUES}
    rng = np.random.default_rng(SUBSAMPLE_SEED)

    for loo_dir in loo_dirs:
        loo_subject = loo_subject_from_dir(loo_dir)
        print(f"\n{'─' * 60}")
        print(f"  Processing fold: {os.path.basename(loo_dir)}  (LOO subject: {loo_subject})")
        print(f"{'─' * 60}")

        csv_files = sorted(glob.glob(os.path.join(loo_dir, "*_scores_*.csv")))
        if not csv_files:
            print(f"  WARNING: No score CSVs found in {loo_dir}, skipping.")
            continue

        healthy_files = []
        tumor_files = []

        for f in csv_files:
            kind = classify_subject(f)
            if kind == "healthy":
                healthy_files.append(f)
            elif kind == "tumor":
                tumor_files.append(f)

        if not healthy_files:
            print(f"  WARNING: No healthy files found, skipping.")
            continue
        if not tumor_files:
            print(f"  WARNING: No tumor files found, skipping.")
            continue

        healthy_scores_all = []
        healthy_names = []
        for hf in healthy_files:
            healthy_scores_all.append(load_subject_scores(hf).flatten())
            healthy_names.append(subject_name_from_file(hf))

        tumor_scores_all = []
        tumor_names = []
        for tf in tumor_files:
            tumor_scores_all.append(load_subject_scores(tf).flatten())
            tumor_names.append(subject_name_from_file(tf))

        print(f"  Healthy: {len(healthy_names)} patients ({', '.join(healthy_names)})")
        print(f"  Tumors: {len(tumor_names)} patients ({', '.join(tumor_names)})")

        for b in range(N_BOOTSTRAP):
            for n_reads in N_VALUES:
                # Sub-sample healthy
                h_subs = []
                for h_scores in healthy_scores_all:
                    n_h = len(h_scores)
                    if n_reads >= n_h:
                        h_subs.append(h_scores)
                    else:
                        idx_h = rng.choice(n_h, size=n_reads, replace=False)
                        h_subs.append(h_scores[idx_h])
                
                # Sub-sample tumor
                t_subs = []
                for t_scores in tumor_scores_all:
                    n_t = len(t_scores)
                    if n_reads >= n_t:
                        t_subs.append(t_scores)
                    else:
                        idx_t = rng.choice(n_t, size=n_reads, replace=False)
                        t_subs.append(t_scores[idx_t])

                # Evaluate all aggregators
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


# ── Plotting ──────────────────────────────────────────────────────────

def plot_heatmap(
    mean_matrix: np.ndarray,
    std_matrix: np.ndarray,
    output_path: str,
    title: str = "AUC (Mann-Whitney U) by Aggregation × N Reads (Pooled All Healthy)",
) -> None:
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

    fig, ax = plt.subplots(figsize=(12, 7))
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
        annot_kws={"fontsize": 10, "fontweight": "bold"},
    )

    ax.set_xlabel("Read per Patient", fontsize=13, labelpad=10)
    ax.set_ylabel("Score Aggregation Method", fontsize=13, labelpad=10)

    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.5)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)

    plt.title(title, fontsize=14, pad=15)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nHeatmap saved to: {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Heatmap of AUC using pooled scores from ALL healthy patients.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Path containing the LOO_* fold directories.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for the heatmap PNG.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Custom plot title.",
    )
    args = parser.parse_args()

    loo_dirs = sorted(glob.glob(os.path.join(args.results_dir, "LOO_*")))
    if not loo_dirs:
        print(f"ERROR: No LOO_* directories found in {args.results_dir}")
        sys.exit(1)

    print(f"Found {len(loo_dirs)} LOO fold(s): {[os.path.basename(d) for d in loo_dirs]}")

    mean_matrix, std_matrix = compute_heatmap_data(loo_dirs)

    print(f"\n{'=' * 80}")
    print("  AUC Results (mean ± std across bootstrap iterations and folds)")
    print(f"{'=' * 80}")
    header = f"{'Aggregator':<20s}" + "".join(f"{'N=' + str(n):>15s}" for n in N_VALUES)
    print(header)
    print("─" * len(header))
    for i, agg_name in enumerate(AGG_ORDER):
        row = f"{agg_name:<20s}"
        for j in range(len(N_VALUES)):
            m = mean_matrix[i, j]
            s = std_matrix[i, j]
            if np.isnan(m):
                row += f"{'N/A':>15s}"
            else:
                row += f"{m:.3f}±{s:.3f}".rjust(15)
        print(row)
    print(f"{'=' * 80}")

    output_path = args.output or os.path.join(args.results_dir, "heatmap_aggregation_auc_pooled_all_healthy.png")
    plot_heatmap(mean_matrix, std_matrix, output_path, title=args.title)


if __name__ == "__main__":
    main()
