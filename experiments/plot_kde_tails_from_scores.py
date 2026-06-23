"""
plot_kde_tails_from_scores.py
Loads per-subject anomaly-score CSVs produced by run_loo_multiround_nystrom.py,
pools rounds, fits KDEs, and produces:

  1. Full-range KDE overlay: all healthy vs all tumor patients
  2. Per-patient KDE vs healthy-baseline overlay
  3. Tail-zoom panels (right tail, left tail) to highlight distributional
     differences in the extremes

Usage
-----
    python experiments/plot_kde_tails_from_scores.py \
        --scores-dir results/loo_multiround_nystrom/m_1/k_6/LOO_Healthy_7

    # Custom tail percentiles:
    python experiments/plot_kde_tails_from_scores.py \
        --scores-dir results/loo_multiround_nystrom/m_1/k_6/LOO_Healthy_7 \
        --right-tail-pct 95 --left-tail-pct 5
"""

import argparse
import csv
import glob
import os
import re
import sys

# ── Path setup ─────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from experiments.experiments_utils import setup_logger

logger = setup_logger(__name__)


# ─────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────

def _load_subject_scores(csv_path: str) -> np.ndarray:
    """Load a multiround score CSV and pool all rounds into one array."""
    all_vals: list[float] = []
    with open(csv_path, "r") as fh:
        reader = csv.reader(fh)
        next(reader)  # skip header
        for row in reader:
            for cell in row:
                cell = cell.strip()
                if cell:
                    all_vals.append(float(cell))
    return np.array(all_vals)


def _subject_name_from_path(csv_path: str) -> str:
    """Extract e.g. 'Healthy_2' from 'Healthy_2_scores_seed42.csv'."""
    basename = os.path.basename(csv_path)
    match = re.match(r"^(\w+_\d+)_scores", basename)
    return match.group(1) if match else os.path.splitext(basename)[0]


def _is_tumor(name: str) -> bool:
    return not name.lower().startswith("healthy")


def _sort_key(name: str):
    """Sort key: healthy first, then by number."""
    prefix = 0 if name.lower().startswith("healthy") else 1
    match = re.search(r"(\d+)", name)
    num = int(match.group(1)) if match else float("inf")
    return (prefix, num)


def _fit_kde(data: np.ndarray) -> gaussian_kde:
    """Fit a Gaussian KDE, adding jitter if variance is zero."""
    valid = data[~np.isnan(data)]
    if valid.size < 2 or np.ptp(valid) == 0:
        valid = valid + np.random.normal(0, 1e-6, size=valid.size)
    return gaussian_kde(valid)


# ─────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────

# Color palette — distinguishable colors for individual patients
_HEALTHY_COLORS = [
    "#2196F3",  # blue
    "#4CAF50",  # green
    "#00BCD4",  # cyan
    "#9C27B0",  # purple
    "#FF9800",  # orange
    "#607D8B",  # blue-grey
]
_TUMOR_COLORS = [
    "#F44336",  # red
    "#E91E63",  # pink
    "#FF5722",  # deep orange
]
_HEALTHY_BASE = "#1565C0"
_TUMOR_BASE = "#C62828"


def _plot_full_overlay(
    subjects: dict[str, np.ndarray],
    out_dir: str,
    n_grid: int = 2048,
) -> str:
    """Full-range KDE overlay: all healthy pooled vs all tumor pooled,
    plus individual per-patient curves in lighter tones."""

    healthy_arrays = [v for k, v in subjects.items() if not _is_tumor(k)]
    tumor_arrays = [v for k, v in subjects.items() if _is_tumor(k)]

    healthy_pool = np.concatenate(healthy_arrays)
    tumor_pool = np.concatenate(tumor_arrays)
    all_data = np.concatenate([healthy_pool, tumor_pool])

    x_min = np.min(all_data)
    x_max = np.max(all_data)
    pad = 0.05 * (x_max - x_min)
    xs = np.linspace(x_min - pad, x_max + pad, n_grid)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Individual patient curves (thin, translucent)
    healthy_names = sorted([k for k in subjects if not _is_tumor(k)], key=_sort_key)
    tumor_names = sorted([k for k in subjects if _is_tumor(k)], key=_sort_key)

    for idx, name in enumerate(healthy_names):
        kde = _fit_kde(subjects[name])
        color = _HEALTHY_COLORS[idx % len(_HEALTHY_COLORS)]
        ax.plot(xs, kde(xs), color=color, alpha=0.35, lw=1.0, label=name)

    for idx, name in enumerate(tumor_names):
        kde = _fit_kde(subjects[name])
        color = _TUMOR_COLORS[idx % len(_TUMOR_COLORS)]
        ax.plot(xs, kde(xs), color=color, alpha=0.35, lw=1.0, label=name)

    # Pooled curves (thick)
    kde_h = _fit_kde(healthy_pool)
    kde_t = _fit_kde(tumor_pool)
    ax.plot(xs, kde_h(xs), color=_HEALTHY_BASE, lw=2.5, label="All Healthy (pooled)")
    ax.plot(xs, kde_t(xs), color=_TUMOR_BASE, lw=2.5, label="All Tumor (pooled)")

    ax.set_xlabel("Anomaly Score (higher = more anomalous)")
    ax.set_ylabel("Density")
    ax.set_title("Anomaly-Score KDEs — Healthy vs Tumor (Full Range)")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.8)
    fig.tight_layout()

    path = os.path.join(out_dir, "kde_full_overlay.pdf")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _plot_per_patient(
    subjects: dict[str, np.ndarray],
    out_dir: str,
    n_grid: int = 2048,
) -> list[str]:
    """Per-patient KDE vs leave-one-out healthy baseline."""
    per_dir = os.path.join(out_dir, "per_patient")
    os.makedirs(per_dir, exist_ok=True)

    all_names = sorted(subjects.keys(), key=_sort_key)
    healthy_names = [n for n in all_names if not _is_tumor(n)]
    paths: list[str] = []

    for name in all_names:
        scores = subjects[name]
        is_sick = _is_tumor(name)

        # Baseline: all healthy EXCEPT this patient (if it's healthy)
        bg_arrays = [subjects[n] for n in healthy_names if n != name]
        if not bg_arrays:
            continue
        bg_pool = np.concatenate(bg_arrays)

        combined = np.concatenate([scores, bg_pool])
        x_min, x_max = combined.min(), combined.max()
        pad = 0.05 * (x_max - x_min)
        xs = np.linspace(x_min - pad, x_max + pad, n_grid)

        fig, ax = plt.subplots(figsize=(10, 5))

        # Baseline
        kde_bg = _fit_kde(bg_pool)
        ax.fill_between(xs, kde_bg(xs), alpha=0.15, color="steelblue")
        ax.plot(xs, kde_bg(xs), color="steelblue", lw=1.5, ls="--",
                label="Healthy Baseline (LOO)")

        # Patient
        p_color = _TUMOR_BASE if is_sick else _HEALTHY_BASE
        kde_p = _fit_kde(scores)
        ax.plot(xs, kde_p(xs), color=p_color, lw=2.0, label=f"{name} KDE")
        ax.fill_between(xs, kde_p(xs), alpha=0.10, color=p_color)

        status = "TUMOR" if is_sick else "HEALTHY"
        ax.set_title(
            f"{name} [{status}] vs Healthy Baseline  —  "
            f"mean={np.mean(scores):.1f}  std={np.std(scores):.1f}"
        )
        ax.set_xlabel("Anomaly Score")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
        fig.tight_layout()

        safe = re.sub(r"[^\w\-.]", "_", name)
        path = os.path.join(per_dir, f"{safe}_kde.pdf")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths.append(path)

    return paths


def _plot_tail_zoom(
    subjects: dict[str, np.ndarray],
    out_dir: str,
    right_pct: float = 95.0,
    left_pct: float = 5.0,
    n_grid: int = 2048,
) -> list[str]:
    """Two-panel figure: zoomed into the right tail and left tail.

    The right tail is the region above the `right_pct` percentile of
    the pooled healthy scores.  The left tail is below `left_pct`.
    This is where tumor patients should diverge the most.
    """
    healthy_names = sorted([k for k in subjects if not _is_tumor(k)], key=_sort_key)
    tumor_names = sorted([k for k in subjects if _is_tumor(k)], key=_sort_key)

    healthy_pool = np.concatenate([subjects[n] for n in healthy_names])
    tumor_pool = np.concatenate([subjects[n] for n in tumor_names])
    all_pool = np.concatenate([healthy_pool, tumor_pool])

    right_thresh = np.percentile(healthy_pool, right_pct)
    left_thresh = np.percentile(healthy_pool, left_pct)

    # ---------- Helper to draw one zoom panel ----------
    def _draw_panel(ax, x_lo, x_hi, title_suffix):
        pad = 0.05 * (x_hi - x_lo) if x_hi > x_lo else 1.0
        xs = np.linspace(x_lo - pad, x_hi + pad, n_grid)

        # Individual patient curves
        for idx, name in enumerate(healthy_names):
            kde = _fit_kde(subjects[name])
            c = _HEALTHY_COLORS[idx % len(_HEALTHY_COLORS)]
            ax.plot(xs, kde(xs), color=c, alpha=0.45, lw=1.0, label=name)

        for idx, name in enumerate(tumor_names):
            kde = _fit_kde(subjects[name])
            c = _TUMOR_COLORS[idx % len(_TUMOR_COLORS)]
            ax.plot(xs, kde(xs), color=c, alpha=0.45, lw=1.0, label=name)

        # Pooled
        kde_h = _fit_kde(healthy_pool)
        kde_t = _fit_kde(tumor_pool)
        ax.plot(xs, kde_h(xs), color=_HEALTHY_BASE, lw=2.5, label="Healthy (pooled)")
        ax.plot(xs, kde_t(xs), color=_TUMOR_BASE, lw=2.5, label="Tumor (pooled)")

        ax.set_xlim(x_lo - pad, x_hi + pad)
        ax.set_xlabel("Anomaly Score")
        ax.set_ylabel("Density")
        ax.set_title(title_suffix)

    paths: list[str] = []

    # --- Right tail ---
    fig, ax = plt.subplots(figsize=(12, 5))
    r_lo = right_thresh
    r_hi = np.max(all_pool)
    _draw_panel(ax, r_lo, r_hi,
                f"Right Tail Zoom (≥ P{right_pct:.0f} of healthy = {right_thresh:.1f})")
    ax.legend(fontsize=7, loc="upper right", framealpha=0.8)
    fig.tight_layout()
    p = os.path.join(out_dir, "kde_right_tail_zoom.pdf")
    fig.savefig(p, dpi=200)
    plt.close(fig)
    paths.append(p)

    # --- Left tail ---
    fig, ax = plt.subplots(figsize=(12, 5))
    l_lo = np.min(all_pool)
    l_hi = left_thresh
    _draw_panel(ax, l_lo, l_hi,
                f"Left Tail Zoom (≤ P{left_pct:.0f} of healthy = {left_thresh:.1f})")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.8)
    fig.tight_layout()
    p = os.path.join(out_dir, "kde_left_tail_zoom.pdf")
    fig.savefig(p, dpi=200)
    plt.close(fig)
    paths.append(p)

    # --- Combined two-panel figure ---
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(20, 6))
    _draw_panel(ax_left, l_lo, l_hi,
                f"Left Tail (≤ P{left_pct:.0f} = {left_thresh:.1f})")
    _draw_panel(ax_right, r_lo, r_hi,
                f"Right Tail (≥ P{right_pct:.0f} = {right_thresh:.1f})")
    # Put legend only on right panel to avoid clutter
    ax_right.legend(fontsize=7, loc="upper right", framealpha=0.8)
    fig.suptitle("Tail-Zoom KDEs — Healthy vs Tumor", fontsize=14, y=1.01)
    fig.tight_layout()
    p = os.path.join(out_dir, "kde_tails_combined.pdf")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    return paths


def _plot_tail_exceedance(
    subjects: dict[str, np.ndarray],
    out_dir: str,
    n_thresholds: int = 500,
) -> str:
    """Complementary CDF (survival function) plot.

    Shows P(X > t) for each patient — the fraction of sequences exceeding
    threshold t.  Tumor patients with heavier right tails will have curves
    above the healthy ones at high thresholds.
    """
    healthy_names = sorted([k for k in subjects if not _is_tumor(k)], key=_sort_key)
    tumor_names = sorted([k for k in subjects if _is_tumor(k)], key=_sort_key)

    all_data = np.concatenate(list(subjects.values()))
    ts = np.linspace(np.min(all_data), np.max(all_data), n_thresholds)

    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, name in enumerate(healthy_names):
        s = subjects[name]
        surv = np.array([np.mean(s > t) for t in ts])
        c = _HEALTHY_COLORS[idx % len(_HEALTHY_COLORS)]
        ax.plot(ts, surv, color=c, lw=1.2, alpha=0.7, label=name)

    for idx, name in enumerate(tumor_names):
        s = subjects[name]
        surv = np.array([np.mean(s > t) for t in ts])
        c = _TUMOR_COLORS[idx % len(_TUMOR_COLORS)]
        ax.plot(ts, surv, color=c, lw=1.2, alpha=0.7, label=name)

    # Pooled
    healthy_pool = np.concatenate([subjects[n] for n in healthy_names])
    tumor_pool = np.concatenate([subjects[n] for n in tumor_names])
    ax.plot(ts, [np.mean(healthy_pool > t) for t in ts],
            color=_HEALTHY_BASE, lw=2.5, label="Healthy (pooled)")
    ax.plot(ts, [np.mean(tumor_pool > t) for t in ts],
            color=_TUMOR_BASE, lw=2.5, label="Tumor (pooled)")

    ax.set_xlabel("Anomaly Score Threshold (t)")
    ax.set_ylabel("P(Score > t)")
    ax.set_title("Tail Exceedance (Survival Function) — Per Patient")
    ax.set_yscale("log")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(out_dir, "tail_exceedance.pdf")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _print_tail_statistics(subjects: dict[str, np.ndarray], right_pct: float, left_pct: float):
    """Print per-patient tail statistics to help interpret the KDEs."""
    healthy_names = sorted([k for k in subjects if not _is_tumor(k)], key=_sort_key)
    tumor_names = sorted([k for k in subjects if _is_tumor(k)], key=_sort_key)

    healthy_pool = np.concatenate([subjects[n] for n in healthy_names])
    right_thresh = np.percentile(healthy_pool, right_pct)
    left_thresh = np.percentile(healthy_pool, left_pct)

    logger.info("")
    logger.info("=" * 80)
    logger.info(" TAIL STATISTICS")
    logger.info("=" * 80)
    logger.info(f"  Healthy right-tail threshold (P{right_pct:.0f}): {right_thresh:.2f}")
    logger.info(f"  Healthy left-tail threshold  (P{left_pct:.0f}):  {left_thresh:.2f}")
    logger.info("")
    logger.info(
        f"{'Subject':<16s} {'Label':<14s} {'Mean':>10s} {'Std':>10s} "
        f"{'%>P{0:.0f}'.format(right_pct):>8s} {'%<P{0:.0f}'.format(left_pct):>8s} "
        f"{'Skewness':>10s} {'Kurtosis':>10s}"
    )
    logger.info("-" * 80)

    from scipy.stats import skew, kurtosis as kurt

    for name in healthy_names + tumor_names:
        s = subjects[name]
        label = "TUMOR" if _is_tumor(name) else "HEALTHY"
        pct_right = 100.0 * np.mean(s > right_thresh)
        pct_left = 100.0 * np.mean(s < left_thresh)
        logger.info(
            f"{name:<16s} {label:<14s} {np.mean(s):>10.2f} {np.std(s):>10.2f} "
            f"{pct_right:>7.2f}% {pct_left:>7.2f}% "
            f"{skew(s):>10.4f} {kurt(s):>10.4f}"
        )
    logger.info("=" * 80)


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="KDE tail analysis of anomaly scores from the Nyström multiround experiment."
    )
    parser.add_argument(
        "--scores-dir", required=True,
        help="Directory containing per-subject *_scores_seed*.csv files.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory for plots (default: <scores-dir>/kde_plots/).",
    )
    parser.add_argument(
        "--right-tail-pct", type=float, default=95.0,
        help="Right tail percentile threshold based on healthy pool (default: 95).",
    )
    parser.add_argument(
        "--left-tail-pct", type=float, default=5.0,
        help="Left tail percentile threshold based on healthy pool (default: 5).",
    )
    parser.add_argument(
        "--seed-filter", type=int, default=None,
        help="If set, only load CSVs matching this seed (e.g. 42).",
    )
    args = parser.parse_args()

    # ── Discover score CSVs ──────────────────────────────────────────
    pattern = os.path.join(args.scores_dir, "*_scores_seed*.csv")
    csv_files = sorted(glob.glob(pattern))
    if args.seed_filter is not None:
        csv_files = [f for f in csv_files if f"seed{args.seed_filter}" in f]

    if not csv_files:
        logger.error(f"No score CSVs found matching {pattern}")
        sys.exit(1)

    logger.info(f"Found {len(csv_files)} score file(s) in {args.scores_dir}")

    # ── Load all subjects ────────────────────────────────────────────
    subjects: dict[str, np.ndarray] = {}
    for csv_path in csv_files:
        name = _subject_name_from_path(csv_path)
        scores = _load_subject_scores(csv_path)
        subjects[name] = scores
        label = "TUMOR" if _is_tumor(name) else "HEALTHY"
        logger.info(f"  {name:<16s} [{label}]  {len(scores):>7,d} scores  "
                     f"mean={np.mean(scores):.2f}  std={np.std(scores):.2f}")

    if not subjects:
        logger.error("No subjects loaded.")
        sys.exit(1)

    # ── Output dir ───────────────────────────────────────────────────
    out_dir = args.output_dir or os.path.join(args.scores_dir, "kde_plots")
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"\nOutput directory: {out_dir}\n")

    # ── Generate plots ───────────────────────────────────────────────
    # 1. Full overlay
    p = _plot_full_overlay(subjects, out_dir)
    logger.info(f"  Saved full overlay:        {p}")

    # 2. Per-patient KDE vs baseline
    paths = _plot_per_patient(subjects, out_dir)
    for p in paths:
        logger.info(f"  Saved per-patient KDE:     {p}")

    # 3. Tail zooms
    tail_paths = _plot_tail_zoom(
        subjects, out_dir,
        right_pct=args.right_tail_pct,
        left_pct=args.left_tail_pct,
    )
    for p in tail_paths:
        logger.info(f"  Saved tail zoom:           {p}")

    # 4. Tail exceedance (survival function)
    p = _plot_tail_exceedance(subjects, out_dir)
    logger.info(f"  Saved tail exceedance:     {p}")

    # 5. Print tail statistics
    _print_tail_statistics(subjects, args.right_tail_pct, args.left_tail_pct)

    logger.info(f"\nAll plots saved to {out_dir}")


if __name__ == "__main__":
    main()
