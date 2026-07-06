import argparse
import glob
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── Path setup ────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

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

def extract_aucs_for_dir(results_dir: str, nu_label: str, n_reads: int, df_records: list):
    loo_dirs = sorted(glob.glob(os.path.join(results_dir, "LOO_*")))
    rng = np.random.default_rng(SUBSAMPLE_SEED)

    for loo_dir in loo_dirs:
        loo_subject = loo_subject_from_dir(loo_dir)
        csv_files = sorted(glob.glob(os.path.join(loo_dir, "*_scores_*.csv")))
        
        test_healthy_file = None
        tumor_files = []
        for f in csv_files:
            subj = subject_name_from_file(f)
            kind = classify_subject(f)
            if kind == "healthy" and subj == loo_subject:
                test_healthy_file = f
            elif kind == "tumor":
                tumor_files.append(f)

        if test_healthy_file is None or not tumor_files:
            continue

        # Single round script: expects 1D scores
        healthy_scores = load_subject_scores(test_healthy_file)
        n_seqs = len(healthy_scores)

        tumor_scores_all = []
        for tf in tumor_files:
            ts = load_subject_scores(tf)
            tumor_scores_all.append(ts)

        if n_reads >= n_seqs:
            h_sub = healthy_scores
            t_subs = tumor_scores_all
        else:
            idx_h = rng.choice(n_seqs, size=n_reads, replace=False)
            h_sub = healthy_scores[idx_h]
            t_subs = []
            for t_scores in tumor_scores_all:
                n_t = len(t_scores)
                idx_t = rng.choice(n_t, size=min(n_reads, n_t), replace=False)
                t_subs.append(t_scores[idx_t])

        for agg_name in AGG_ORDER:
            agg_fn = AGGREGATORS[agg_name]
            h_agg = np.array([agg_fn(h_sub)])
            t_aggs = np.array([agg_fn(ts) for ts in t_subs])
            auc = auc_healthy_vs_sick(h_agg, t_aggs)
            
            df_records.append({
                "AUC": auc,
                "Aggregator": agg_name,
                "Nu": nu_label
            })

def main():
    parser = argparse.ArgumentParser(description="Plot Grouped Boxplots of AUCs across Nu values (Single Round / Folds)")
    parser.add_argument("--results-dirs", nargs="+", required=True, help="List of single-round cached result directories")
    parser.add_argument("--nu-labels", nargs="+", required=True, help="List of Nu labels corresponding to results-dirs")
    parser.add_argument("--n-reads", type=int, default=50000, help="Fixed number of reads to subsample (default: 50000)")
    parser.add_argument("--output", default="boxplot_nu_comparison_folds.png", help="Output PNG path")
    args = parser.parse_args()

    if len(args.results_dirs) != len(args.nu_labels):
        print("ERROR: --results-dirs and --nu-labels must have the same number of arguments")
        sys.exit(1)

    df_records = []
    for res_dir, label in zip(args.results_dirs, args.nu_labels):
        print(f"Extracting AUCs for Nu={label} from {res_dir} ...")
        extract_aucs_for_dir(res_dir, label, args.n_reads, df_records)

    df = pd.DataFrame(df_records)
    if df.empty:
        print("No data extracted. Check your results-dirs.")
        sys.exit(1)

    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid")
    
    # Create grouped boxplot. With 7 points, whis=(5, 95) effectively acts as min/max.
    ax = sns.boxplot(
        data=df, 
        x="Aggregator", 
        y="AUC", 
        hue="Nu", 
        order=AGG_ORDER,
        whis=(5, 95),
        palette="Set2",
        width=0.6
    )

    ax.set_title(f"AUC Distributions Across 7 Folds (N={args.n_reads} sequences)", fontsize=16, pad=15)
    ax.set_ylabel("AUC (Mann-Whitney U)", fontsize=14)
    ax.set_xlabel("Aggregation Method", fontsize=14)
    
    # Rotate x labels slightly
    plt.xticks(rotation=20, ha="right", fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.legend(title="Nu Value", title_fontsize=13, fontsize=12, bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    
    plt.savefig(args.output, dpi=300)
    print(f"\nPlot saved to {args.output}")

if __name__ == "__main__":
    main()
