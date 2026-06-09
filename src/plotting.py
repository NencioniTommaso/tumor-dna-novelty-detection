"""
plotting.py
Visualization functions for anomaly score distributions.
Generates per-patient KDE plots and combined healthy-vs-tumor overlays.

Isolates the heavy matplotlib dependency from core evaluation logic.
"""

import logging
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

logger = logging.getLogger(__name__)


def _patient_sort_key(name: str):
    """Sort key: healthy patients first, then by number."""
    prefix = 0 if name.lower().startswith("healthy") else 1
    match = re.search(r"(\d+)", name)
    num = int(match.group(1)) if match else float("inf")
    return (prefix, num)


def generate_score_distribution_plots(per_patient_data, plot_dir, seed):
    """
    Generates:
      1. One KDE plot per patient (individual score vs Leave-One-Out Healthy Baseline)
      2. One combined overlay (all healthy vs all tumor)

    Parameters
    ----------
    per_patient_data : list of dict
        Each dict has keys: 'short_name', 'label', 'inverted_scores', 'mean_score'.
    plot_dir : str
        Output directory for the plots.
    seed : int
        Random seed used in the experiment (included in filenames).
    """
    os.makedirs(plot_dir, exist_ok=True)
    per_patient_dir = os.path.join(plot_dir, "per_patient")
    os.makedirs(per_patient_dir, exist_ok=True)

    # Sort: healthy first, then tumor
    sorted_data = sorted(per_patient_data, key=lambda d: _patient_sort_key(d['short_name']))

    # --- 1. Per-patient KDE plots with Leave-One-Out Baseline ---
    for pdata in sorted_data:
        fig, ax = plt.subplots(figsize=(8, 5))
        scores = pdata['inverted_scores']
        is_tumor = (pdata['label'] == -1)
        
        # Build the Leave-One-Out healthy baseline
        # Include all healthy patients EXCEPT the current one (if it's healthy)
        bg_arrays = [
            d['inverted_scores'] for d in sorted_data
            if d['label'] != -1 and d['short_name'] != pdata['short_name']
        ]
        
        x_min, x_max = scores.min(), scores.max()
        
        if bg_arrays:
            bg_all = np.concatenate(bg_arrays)
            x_min = min(x_min, bg_all.min())
            x_max = max(x_max, bg_all.max())
            x_grid = np.linspace(x_min, x_max, 500)
            
            # Plot background baseline
            ax.hist(bg_all, bins=100, density=True, alpha=0.3, color='gray', edgecolor='none', label='Healthy Baseline (Others)')
            if len(bg_all) > 2 and np.ptp(bg_all) > 0:
                kde_bg = gaussian_kde(bg_all)
                ax.plot(x_grid, kde_bg(x_grid), color='dimgray', linewidth=1.5, linestyle='--', label='Baseline KDE')
        else:
            x_grid = np.linspace(x_min, x_max, 500)

        # Plot the patient
        patient_color = 'red' if is_tumor else 'green'
        patient_hist_color = 'lightcoral' if is_tumor else 'lightgreen'
        
        ax.hist(scores, bins=100, density=True, alpha=0.5, color=patient_hist_color, edgecolor='none', label='Patient Reads')

        if len(scores) > 2 and np.ptp(scores) > 0:
            kde_p = gaussian_kde(scores)
            ax.plot(x_grid, kde_p(x_grid), color=patient_color, linewidth=2.0, label='Patient KDE')

        status = "TUMOR" if is_tumor else "HEALTHY"
        ax.set_title(f"{pdata['short_name']} [{status}] vs Baseline — Mean Score: {pdata['mean_score']:.4f}", color=patient_color)
        ax.set_xlabel("Anomaly Score (Higher = More Anomalous)")
        ax.set_ylabel("Density")
        if ax.get_legend_handles_labels()[1]:
            ax.legend()
        fig.tight_layout()

        safe_name = re.sub(r"[^\w\-.]", "_", pdata['short_name'])
        patient_subdir = os.path.join(per_patient_dir, safe_name)
        os.makedirs(patient_subdir, exist_ok=True)
        path = os.path.join(patient_subdir, f"{safe_name}_score_distribution_seed{seed}.pdf")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info(f"  Saved per-patient plot: {path}")

    # --- 2. Combined overlay: All Healthy vs All Tumor ---
    healthy_all = np.concatenate([d['inverted_scores'] for d in sorted_data if d['label'] != -1])
    tumor_all = np.concatenate([d['inverted_scores'] for d in sorted_data if d['label'] == -1])

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(healthy_all, bins=100, density=True, alpha=0.3, color='blue', edgecolor='none', label='All Healthy Reads')
    ax.hist(tumor_all, bins=100, density=True, alpha=0.3, color='red', edgecolor='none', label='All Tumor Reads')

    if len(healthy_all) > 2 and np.ptp(healthy_all) > 0:
        kde_h = gaussian_kde(healthy_all)
        x_min = min(healthy_all.min(), tumor_all.min())
        x_max = max(healthy_all.max(), tumor_all.max())
        x_grid = np.linspace(x_min, x_max, 500)
        ax.plot(x_grid, kde_h(x_grid), color='blue', linewidth=2.0, label='Healthy KDE')

    if len(tumor_all) > 2 and np.ptp(tumor_all) > 0:
        kde_t = gaussian_kde(tumor_all)
        x_grid = np.linspace(x_min, x_max, 500)
        ax.plot(x_grid, kde_t(x_grid), color='red', linewidth=2.0, label='Tumor KDE')

    ax.set_title("Distribution of Sequence Anomaly Scores: All Healthy vs All Tumor")
    ax.set_xlabel("Anomaly Score (Higher = More Anomalous)")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()

    path = os.path.join(plot_dir, f"healthy_vs_tumor_overlay_seed{seed}.pdf")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"  Saved combined overlay plot: {path}")


def generate_loo_fold_plot(per_patient_data, plot_dir, fold_name, seed):
    """Generate a single overlay plot for one LOO fold.

    Shows the score distributions of the held-out healthy patient and
    the tumor patient side by side — no baseline since only one healthy
    patient is in the test set.

    Parameters
    ----------
    per_patient_data : list of dict
        Each dict has keys: 'short_name', 'label', 'inverted_scores', 'mean_score'.
        Expected to contain exactly 2 entries (1 healthy, 1 tumor).
    plot_dir : str
        Output directory for the plot.
    fold_name : str
        Fold identifier (e.g. ``"LOO_Healthy_3"``), embedded in the filename.
    seed : int
        Random seed used in the experiment.
    """
    os.makedirs(plot_dir, exist_ok=True)

    # Separate healthy and tumor entries
    healthy = [d for d in per_patient_data if d['label'] != -1]
    tumor = [d for d in per_patient_data if d['label'] == -1]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Determine shared x-axis range
    all_scores = np.concatenate([d['inverted_scores'] for d in per_patient_data])
    x_min, x_max = all_scores.min(), all_scores.max()
    x_grid = np.linspace(x_min, x_max, 500)

    # Plot healthy patient(s)
    for d in healthy:
        scores = d['inverted_scores']
        ax.hist(scores, bins=100, density=True, alpha=0.3, color='blue',
                edgecolor='none', label=f"{d['short_name']} (Healthy)")
        if len(scores) > 2 and np.ptp(scores) > 0:
            kde = gaussian_kde(scores)
            ax.plot(x_grid, kde(x_grid), color='blue', linewidth=2.0,
                    label=f"{d['short_name']} KDE")

    # Plot tumor patient(s)
    for d in tumor:
        scores = d['inverted_scores']
        ax.hist(scores, bins=100, density=True, alpha=0.3, color='red',
                edgecolor='none', label=f"{d['short_name']} (Tumor)")
        if len(scores) > 2 and np.ptp(scores) > 0:
            kde = gaussian_kde(scores)
            ax.plot(x_grid, kde(x_grid), color='red', linewidth=2.0,
                    label=f"{d['short_name']} KDE")

    # Annotate mean scores
    for d in per_patient_data:
        color = 'red' if d['label'] == -1 else 'blue'
        ax.axvline(d['mean_score'], color=color, linestyle=':', alpha=0.7)

    ax.set_title(f"{fold_name}: Held-Out Healthy vs Tumor", fontsize=13)
    ax.set_xlabel("Anomaly Score (Higher = More Anomalous)")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()

    path = os.path.join(plot_dir, f"{fold_name}_overlay_seed{seed}.pdf")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"  Saved LOO fold plot: {path}")
