"""
compute_oa_from_scores.py
Reads the per-fold anomaly-score CSVs (two columns: healthy patient vs. tumor
patient) and computes the Overlapping Area (OA) between the two distributions
using the existing KDE utilities from src/oa/evaluation_oa.py.

Usage
-----
    python experiments/compute_oa_from_scores.py \
        --scores-dir results/loo/m_1/k_6/Colo_11/anomaly_scores

The script will:
  1. Load every CSV in the scores directory.
  2. For each CSV, fit KDEs on both columns and compute OA (healthy vs. tumor).
  3. Compute pairwise OAs between all healthy patients across CSVs.
  4. Print summary tables and save a results CSV alongside the scores.
"""

import argparse
import csv
import glob
import itertools
import os
import sys

import numpy as np

# Resolve project root so imports work from any CWD
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from src.oa.evaluation_oa import compute_kde, compute_oa


def load_score_csv(path: str):
    """Read a two-column anomaly-score CSV.

    Returns
    -------
    col1_name, col2_name : str
        Header names (e.g. "Healthy_2", "Colo_11").
    col1, col2 : np.ndarray
        Anomaly scores for each patient.
    """
    with open(path, "r") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        col1_name = header[0].strip()
        col2_name = header[1].strip()
        col1, col2 = [], []
        for row in reader:
            if row[0]:
                col1.append(float(row[0]))
            if len(row) > 1 and row[1]:
                col2.append(float(row[1]))
    return col1_name, col2_name, np.array(col1), np.array(col2)


def compute_oa_for_pair(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """Fit KDEs to two score arrays and return the Overlapping Area."""
    # Use a shared evaluation grid covering both distributions
    all_scores = np.concatenate([scores_a, scores_b])
    xmax = np.nanmax(np.abs(all_scores)) * 1.1  # slight padding
    xmin = np.nanmin(all_scores) - 0.1 * np.ptp(all_scores)

    num_points = 1024
    xs = np.linspace(xmin, xmax, num_points)

    # Fit KDEs
    kde_a = _fit_kde(scores_a)
    kde_b = _fit_kde(scores_b)

    ya = kde_a(xs)
    yb = kde_b(xs)

    # OA = 1 - (1/2) * integral |f_a - f_b| dx
    area_diff = np.trapezoid(np.abs(ya - yb), xs)
    oa = 1.0 - area_diff / 2.0
    return oa


def _fit_kde(data: np.ndarray):
    """Fit a Gaussian KDE, adding tiny jitter if variance is zero."""
    from scipy.stats import gaussian_kde

    valid = data[~np.isnan(data)]
    if valid.size < 2 or np.ptp(valid) == 0:
        valid = valid + np.random.normal(0, 1e-6, size=valid.size)
    return gaussian_kde(valid)


def main():
    parser = argparse.ArgumentParser(
        description="Compute Overlapping Area (OA) from saved anomaly-score CSVs."
    )
    parser.add_argument(
        "--scores-dir",
        required=True,
        help="Directory containing the per-fold anomaly-score CSVs.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Path for the output CSV (default: <scores-dir>/oa_results.csv).",
    )
    args = parser.parse_args()

    csv_files = sorted(glob.glob(os.path.join(args.scores_dir, "*.csv")))
    # Exclude our own output file from previous runs
    csv_files = [f for f in csv_files if os.path.basename(f) != "oa_results.csv"]
    if not csv_files:
        print(f"ERROR: No CSV files found in {args.scores_dir}")
        sys.exit(1)

    print(f"Found {len(csv_files)} score file(s) in {args.scores_dir}\n")

    # ------------------------------------------------------------------
    # 1. Healthy vs. Tumor OA (one per CSV)
    # ------------------------------------------------------------------
    print("=" * 70)
    print("  HEALTHY vs. TUMOR")
    print("=" * 70)

    results = []
    healthy_scores_map = {}  # name -> scores array (collected for step 2)

    for csv_path in csv_files:
        basename = os.path.basename(csv_path)
        col1_name, col2_name, scores1, scores2 = load_score_csv(csv_path)
        oa = compute_oa_for_pair(scores1, scores2)
        anomaly_score = 1.0 - oa

        results.append({
            "comparison_type": "healthy_vs_tumor",
            "patient_a": col1_name,
            "patient_b": col2_name,
            "n_scores_a": len(scores1),
            "n_scores_b": len(scores2),
            "oa": oa,
            "anomaly_score": anomaly_score,
        })

        # Collect healthy patient scores (column 1 is always the healthy patient)
        if col1_name not in healthy_scores_map:
            healthy_scores_map[col1_name] = scores1

        print(f"  {basename:<45s}  {col1_name} vs {col2_name}  ->  OA = {oa:.4f}  (anomaly = {anomaly_score:.4f})")

    oas_ht = np.array([r["oa"] for r in results])
    print(f"\n{'─' * 70}")
    print(f"  Mean OA (healthy vs tumor):  {np.mean(oas_ht):.4f} ± {np.std(oas_ht):.4f}")
    print(f"  Min / Max OA:                {np.min(oas_ht):.4f} / {np.max(oas_ht):.4f}")
    print(f"{'─' * 70}")

    # ------------------------------------------------------------------
    # 2. Healthy vs. Healthy OA (all pairwise combinations)
    # ------------------------------------------------------------------
    healthy_names = sorted(healthy_scores_map.keys())
    if len(healthy_names) >= 2:
        print(f"\n{'=' * 70}")
        print("  HEALTHY vs. HEALTHY (pairwise)")
        print(f"{'=' * 70}")

        hh_results = []
        for name_a, name_b in itertools.combinations(healthy_names, 2):
            oa = compute_oa_for_pair(healthy_scores_map[name_a], healthy_scores_map[name_b])
            anomaly_score = 1.0 - oa

            row = {
                "comparison_type": "healthy_vs_healthy",
                "patient_a": name_a,
                "patient_b": name_b,
                "n_scores_a": len(healthy_scores_map[name_a]),
                "n_scores_b": len(healthy_scores_map[name_b]),
                "oa": oa,
                "anomaly_score": anomaly_score,
            }
            hh_results.append(row)
            results.append(row)

            print(f"  {name_a} vs {name_b}  ->  OA = {oa:.4f}  (anomaly = {anomaly_score:.4f})")

        oas_hh = np.array([r["oa"] for r in hh_results])
        print(f"\n{'─' * 70}")
        print(f"  Mean OA (healthy vs healthy):  {np.mean(oas_hh):.4f} ± {np.std(oas_hh):.4f}")
        print(f"  Min / Max OA:                  {np.min(oas_hh):.4f} / {np.max(oas_hh):.4f}")
        print(f"{'─' * 70}")

        # Compare the two groups
        print(f"\n{'=' * 70}")
        print("  COMPARISON SUMMARY")
        print(f"{'=' * 70}")
        print(f"  Healthy vs Tumor  — mean OA = {np.mean(oas_ht):.4f} ± {np.std(oas_ht):.4f}")
        print(f"  Healthy vs Healthy — mean OA = {np.mean(oas_hh):.4f} ± {np.std(oas_hh):.4f}")
        print(f"  Δ mean OA = {np.mean(oas_hh) - np.mean(oas_ht):.4f}")
        print(f"{'=' * 70}")

    # ------------------------------------------------------------------
    # Save all results
    # ------------------------------------------------------------------
    out_path = args.output_csv or os.path.join(args.scores_dir, "oa_results.csv")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fieldnames = ["comparison_type", "patient_a", "patient_b", "n_scores_a", "n_scores_b", "oa", "anomaly_score"]
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
