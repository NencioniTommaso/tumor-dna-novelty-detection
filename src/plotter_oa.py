"""
plotter_oa.py
Plotting utilities for the Overlapping Area (OA) KDE methodology.
"""

import os
import re
from collections import defaultdict
from itertools import cycle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------- Color cycle for fill_between ----------
_fill_colors = cycle(plt.cm.Set2.colors)


def init_overlapping_plot(
    xs,
    y_intra,
    y_inter,
    metric_name,
    sample_size,
    OA=None,
    colors=None,
    label1="Intra",
    label2=None,
    title=None,
    plt_overlap=True,
    line_width=0.5,
):
    """
    Create a plot with one or two KDE curves.
    """
    OA_label = None
    if OA is not None:
        OA_label = f"Overlap (OA = {OA:.3f})"

    fig, ax = plt.subplots(figsize=(7, 5))

    fill_color = next(_fill_colors)

    if y_inter is None:
        # Single curve
        if colors is not None:
            ax.plot(xs, y_intra, label=label1, lw=line_width * 2, color=colors[0])
        else:
            ax.plot(xs, y_intra, label=label1, lw=line_width)
    else:
        if colors is not None:
            color_intra, color_inter = colors
            ax.plot(xs, y_intra, label=label1, lw=line_width * 4, color=color_intra)
            ax.plot(xs, y_inter, label=label2, lw=line_width, color=color_inter)
        else:
            ax.plot(xs, y_intra, label=label1, lw=line_width)
            ax.plot(xs, y_inter, label=label2, lw=line_width)

        if plt_overlap:
            ax.fill_between(
                xs,
                np.minimum(y_intra, y_inter),
                color=fill_color,
                alpha=0.4,
                label=OA_label,
            )

    ax.set_xlabel(f"distance [{metric_name}]")
    ax.set_ylabel("density")
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(f"KDE intra vs inter — sample_size={sample_size}")

    return fig, ax


def add_overlapping_curves(ax, xs, ys, color=None, alpha=None, label=None, line_width=0.5):
    """
    Add an additional curve to an existing plot.
    """
    local_alpha = alpha if alpha is not None else 0.5
    if color is not None:
        ax.plot(xs, ys, lw=line_width, alpha=local_alpha, color=color, label=label)
    else:
        ax.plot(xs, ys, lw=line_width, alpha=local_alpha, label=label)


def _patient_key(name):
    """
    Sort key: healthy patients first, then by number.
    """
    prefix = 0 if name.lower().startswith("healthy") else 1
    match = re.search(r"(\d+)", name)
    num = int(match.group(1)) if match else float("inf")
    return (prefix, num)


def plot_single_patient_oa(
    xs,
    y_intra,
    y_inter,
    oa,
    patient_name,
    metric_name,
    out_dir,
    seed,
    label_intra="Reference (Healthy Intra)",
    label_inter=None,
):
    """
    Plot the two KDE curves and OA shading for a single patient.
    """
    if label_inter is None:
        label_inter = f"{patient_name} (Inter)"

    COLOR_INTRA = "blue"
    COLOR_INTER = "red"

    fig, ax = init_overlapping_plot(
        xs,
        y_intra,
        y_inter,
        metric_name=metric_name,
        sample_size="full",
        OA=oa,
        colors=(COLOR_INTRA, COLOR_INTER),
        label1=label_intra,
        label2=label_inter,
        title=f"{patient_name} — OA = {oa:.4f}",
        plt_overlap=True,
        line_width=0.5,
    )

    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    safe_name = re.sub(r"[^\w\-.]", "_", patient_name)
    out_path = os.path.join(out_dir, f"{safe_name}_oa_seed{seed}.pdf")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_all_patients_overlay(
    xs,
    y_intra,
    patient_results,
    metric_name,
    out_dir,
    seed,
    ref_label="reference pdf",
    healthy_label="healthy",
    tumor_label="tumor",
    color_healthy="green",
    color_tumor="red",
):
    """
    Overlay all patient inter-KDE curves on top of the reference intra-KDE.

    Parameters
    ----------
    patient_results : list of dict
        Each dict has keys: 'filename', 'label', 'y_inter', 'oa'
    """
    COLOR_REFERENCE = "blue"

    # Initialize plot with reference curve only
    fig, ax = init_overlapping_plot(
        xs,
        y_intra,
        y_inter=None,
        metric_name=metric_name,
        sample_size="full",
        colors=(COLOR_REFERENCE,),
        label1=ref_label,
        title="Patients pdf against reference pdf",
        plt_overlap=False,
        line_width=1.5,
    )

    healthy_lbl_put = False
    tumor_lbl_put = False

    for result in patient_results:
        is_tumor = result["label"] == -1
        color = color_tumor if is_tumor else color_healthy

        lbl = None
        if is_tumor and not tumor_lbl_put:
            lbl = tumor_label
            tumor_lbl_put = True
        elif not is_tumor and not healthy_lbl_put:
            lbl = healthy_label
            healthy_lbl_put = True

        add_overlapping_curves(ax, xs, result["y_inter"], color=color, alpha=0.6, label=lbl)

    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"all_patients_overlay_seed{seed}.pdf")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_results_patients(patient_results, metric_name, out_dir, seed):
    """
    Bar chart of OA values per patient, sorted healthy-first.
    Mirrors Innocenti's plotter.plot_results_patients, adapted for single
    deterministic OA values (no mean±std needed with the kernel approach).

    Parameters
    ----------
    patient_results : list of dict
        Each dict has keys: 'filename', 'label', 'oa'
    """
    # Sort: healthy first, then by number
    patient_results = sorted(patient_results, key=lambda r: _patient_key(r["filename"]))

    names = [r["filename"] for r in patient_results]
    oas = np.array([r["oa"] for r in patient_results])
    x = np.arange(len(names))

    plt.figure(figsize=(10, 6))

    # Color by label
    colors = ["green" if r["label"] != -1 else "red" for r in patient_results]
    plt.scatter(x, oas, color=colors, s=30, zorder=3)

    # Annotate values
    for xi, oa_val in zip(x, oas):
        plt.text(xi + 0.15, oa_val, f"{oa_val:.4f}", fontsize=8, va="center")

    plt.xticks(x, names, rotation=45, ha="right", fontsize=8)

    # y-axis limits
    data_low = oas.min()
    data_high = oas.max()
    margin = (data_high - data_low) * 0.1
    lower_lim = data_low - margin
    upper_lim = min(1.0, data_high + margin)
    plt.ylim(lower_lim, upper_lim)

    plt.ylabel("OA")
    plt.title(f"OAs with reference distribution [{metric_name}]")
    plt.grid(True, linestyle="--", alpha=0.5)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="green", markersize=6, label="Healthy"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="red", markersize=6, label="Tumor"),
    ]
    leg = plt.legend(
        handles=legend_elements, frameon=True, loc="upper left",
        edgecolor="black", facecolor="white", framealpha=0.5,
    )
    leg.get_frame().set_linewidth(0.5)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"oa_results_patients_seed{seed}.pdf")
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def make_split_plot(patient_results, metric_name, out_dir, seed):
    """
    Split plot: healthy on left, tumor on right.
    """
    patient_results = sorted(patient_results, key=lambda r: _patient_key(r["filename"]))

    names = [r["filename"] for r in patient_results]
    oas = np.array([r["oa"] for r in patient_results])

    healthy_idx = [i for i, r in enumerate(patient_results) if r["label"] != -1]
    sick_idx = [i for i, r in enumerate(patient_results) if r["label"] == -1]

    plt.figure(figsize=(5, 4))

    # Healthy
    for i in healthy_idx:
        plt.plot(0, oas[i], "o", color="green", markersize=5, zorder=3)
        plt.text(0.05, oas[i], names[i], ha="left", va="center",
                 fontsize=5, color="green", weight="bold")

    # Tumor
    for i in sick_idx:
        plt.plot(1, oas[i], "o", color="red", markersize=5, zorder=3)
        plt.text(0.95, oas[i], names[i], ha="right", va="center",
                 fontsize=5, color="red", weight="bold")

    plt.xticks([0, 1], ["Healthy", "Tumor"])
    plt.xlim(-0.5, 1.5)

    data_low = oas.min()
    data_high = oas.max()
    margin = (data_high - data_low) * 0.1
    plt.ylim(data_low - margin, min(1.0, data_high + margin))

    ymin, ymax = plt.ylim()
    step = 0.01
    y_ticks = list(np.arange(round(ymin, 3), round(ymax, 3) + step, step))
    y_ticks = sorted(set(y_ticks))
    plt.yticks(y_ticks, [f"{y:.3f}" for y in y_ticks], fontsize=4)

    for oa_val in oas:
        plt.axhline(oa_val, color="gray", linestyle=":", linewidth=0.5, alpha=0.4)

    plt.ylabel("OA")
    plt.title(f"OAs with reference distribution [{metric_name}]")
    plt.grid(True, linestyle="--", alpha=0.5)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="green", markersize=6, label="Healthy"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="red", markersize=6, label="Tumor"),
    ]
    plt.legend(
        handles=legend_elements, frameon=True, loc="upper left",
        edgecolor="black", facecolor="white", framealpha=0.5, fontsize=6.5,
    )

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"oa_split_plot_seed{seed}.pdf")
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_pairwise_oa_heatmap(oa_matrix, labels, out_dir, seed):
    """
    Plot an annotated heatmap of pairwise OA values between reference distributions.

    Parameters
    ----------
    oa_matrix : np.ndarray, shape (N, N)
        Symmetric matrix where oa_matrix[i,j] = OA between reference i and reference j.
    labels : list[str]
        Short label for each combination, e.g. ["H2+H3+H4", "H2+H3+H5", ...].
    out_dir : str
        Output directory for the plot.
    seed : int
        Seed used (for filename).
    """
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(6, n * 1.5), max(5, n * 1.2)))
    im = ax.imshow(oa_matrix, cmap="YlGnBu", vmin=oa_matrix.min() - 0.01, vmax=1.0)

    # Annotate each cell with its OA value
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{oa_matrix[i, j]:.3f}", ha="center", va="center", fontsize=7)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Pairwise OA Between Reference Distributions")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"reference_pairwise_oa_seed{seed}.pdf")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

