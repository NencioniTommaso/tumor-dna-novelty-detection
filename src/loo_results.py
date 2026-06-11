"""
loo_results.py
Utilities for persisting and summarising Leave-One-Out experiment results.
"""

import csv
import logging
import os
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def save_loo_results(results: List[Dict[str, Any]], output_path: str) -> None:
    """Write per-fold LOO results to a CSV file.

    Parameters
    ----------
    results : list of dict
        Each dict must contain at least:
        ``fold_name``, ``held_out_patient``, ``patient_auc``,
        ``seq_auc``, ``num_train_seqs``, ``num_test_seqs``.
    output_path : str
        Destination CSV path.  Parent directories are created if needed.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fieldnames = [
        "fold_name",
        "held_out_patient",
        "tumor_patient",
        "patient_auc",
        "seq_auc",
        "mean_score_healthy",
        "mean_score_tumor",
        "num_train_seqs",
        "num_test_seqs",
    ]

    with open(output_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    logger.info(f"LOO results saved to {output_path}")


def save_fold_anomaly_scores(
    per_patient_data: List[Dict[str, Any]],
    output_path: str,
) -> None:
    """Write per-sequence anomaly scores for both tested subjects to CSV.

    Creates a CSV with two columns — one per tested patient (held-out
    healthy and tumor).  Column headers are the short patient names
    (e.g. ``Healthy_2``, ``Colo_15``).

    Parameters
    ----------
    per_patient_data : list of dict
        Exactly two entries (healthy + tumor), each containing at least
        ``'short_name'`` and ``'inverted_scores'`` (1-D array).
    output_path : str
        Destination CSV path.  Parent directories are created if needed.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    names = [d["short_name"] for d in per_patient_data]
    scores = [d["inverted_scores"] for d in per_patient_data]
    max_len = max(len(s) for s in scores)

    with open(output_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(names)
        for i in range(max_len):
            row = [
                str(scores[col][i]) if i < len(scores[col]) else ""
                for col in range(len(scores))
            ]
            writer.writerow(row)

    logger.info(f"Fold anomaly scores saved to {output_path}")


def print_loo_summary(results: List[Dict[str, Any]], log: logging.Logger) -> None:
    """Print a formatted summary table and aggregate stats to the logger.

    Parameters
    ----------
    results : list of dict
        Same format as ``save_loo_results``.
    log : logging.Logger
        Logger instance to write to.
    """
    log.info("")
    log.info("=" * 72)
    log.info(" LEAVE-ONE-OUT CROSS-VALIDATION SUMMARY")
    log.info("=" * 72)
    log.info(
        f"{'Fold':<20s} {'Held-Out':<14s} {'Patient AUC':>12s} {'Seq AUC':>10s}"
    )
    log.info("-" * 72)

    for r in results:
        log.info(
            f"{r['fold_name']:<20s} "
            f"{r['held_out_patient']:<14s} "
            f"{r['patient_auc']:>12.4f} "
            f"{r['seq_auc']:>10.4f}"
        )

    aucs = np.array([r["patient_auc"] for r in results])
    log.info("-" * 72)
    log.info(f"{'Mean ± Std':<36s} {np.mean(aucs):>12.4f} ± {np.std(aucs):.4f}")
    log.info(f"{'Min / Max':<36s} {np.min(aucs):>12.4f} / {np.max(aucs):.4f}")
    log.info("=" * 72)
