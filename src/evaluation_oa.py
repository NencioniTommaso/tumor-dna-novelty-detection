import numpy as np
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
import logging

def _process_patient(info, patient_K, xs, y_intra, downsample_kde):
    # Inter-distances: patient sequences vs healthy train sequences
    patient_inter_kernel_vals = patient_K.flatten()
    patient_inter_distances = compute_distances(patient_inter_kernel_vals)
    
    # Ensure we evaluate the patient KDE on the same xs grid
    _, y_inter = compute_kde(patient_inter_distances, xmax=xs[-1], num_points=len(xs), downsample=downsample_kde)
    
    oa = compute_oa(y_intra, y_inter, xs)
    patient_score = 1.0 - oa  # Higher score = more anomalous
    
    return info['label'], patient_score, oa, info['filename'], y_inter

def compute_distances(K: np.ndarray) -> np.ndarray:
    """
    Computes distances given a normalized kernel matrix (where K(x,x)=1).
    D^2(x,y) = K(x,x) + K(y,y) - 2K(x,y) = 2 - 2K(x,y)
    """
    dist_sq = 2.0 * (1.0 - K)
    dist_sq = np.clip(dist_sq, a_min=0, a_max=None)
    return np.sqrt(dist_sq)

def compute_kde(distances: np.ndarray, xmax: float = None, num_points: int = 512, downsample: bool = True, max_samples: int = 2000000):
    """
    Fits a Gaussian KDE to the distances and evaluates it on a grid.
    """
    valid = distances[~np.isnan(distances)]
    
    if downsample and valid.size > max_samples:
        valid = np.random.choice(valid, size=max_samples, replace=False)
        
    if valid.size < 2 or np.ptp(valid) == 0:
        valid = valid + np.random.normal(0, 1e-6, size=valid.size)
        
    kde = gaussian_kde(valid)
    max_val = np.nanmax(valid)
    upper = xmax if xmax is not None else max_val
    xs = np.linspace(0, upper, num_points)
    ys = kde(xs)
    
    return xs, ys

def compute_oa(y_intra: np.ndarray, y_inter: np.ndarray, xs: np.ndarray) -> float:
    """
    Computes Overlapping Area (OA) between two KDE curves.
    """
    area = np.trapezoid(np.abs(y_intra - y_inter), xs)
    OA = 1.0 - area / 2.0
    return OA

def evaluate_patient_level_oa_method(K_train, K_test, test_files_info, logger, n_jobs=-1, downsample_kde=True, plot_dir=None, mismatches=0, max_k=6, seed=42, is_deep=False):
    """
    Computes Overlapping Area (OA) per patient.
    When plot_dir is provided, generates visualizations.
    """
    logger.info("\n--- Patient-Level Anomaly Aggregation (OA Method) ---")
    logger.info("Computing Healthy Intra-Distances and Reference KDE...")

    # 1. Compute intra-distances from K_train (healthy vs healthy)
    N_train = K_train.shape[0]
    
    # Extract strictly the upper triangle (no self-comparisons)
    indices_i, indices_j = np.triu_indices(N_train, k=1)
    train_intra_kernel_vals = K_train[indices_i, indices_j]
    
    train_intra_distances = compute_distances(train_intra_kernel_vals)
    xs, y_intra = compute_kde(train_intra_distances, downsample=downsample_kde)
    
    patient_y_true = []
    patient_scores = []
    
    tasks = []
    current_idx = 0
    for info in test_files_info:
        num_seqs = info['num_sequences']
        patient_K = K_test[current_idx: current_idx + num_seqs, :]
        tasks.append((info, patient_K))
        current_idx += num_seqs
        
    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_patient)(info, patient_K, xs, y_intra, downsample_kde) for info, patient_K in tasks
    )
    
    # Collect results and build data for plotting
    patient_plot_data = []
    for label, score, oa, filename, y_inter in results:
        patient_y_true.append(label)
        patient_scores.append(score)
        
        # Shorten filename (e.g. Colo_6_merged_subset_1200000.fa -> Colo_6)
        parts = filename.split("_")
        short_name = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else filename
        
        patient_plot_data.append({
            'filename': short_name,
            'label': label,
            'oa': oa,
            'y_inter': y_inter,
        })
        
        status = "TUMOR" if label == -1 else "HEALTHY"
        logger.info(f"[{status}] {short_name} -> Overlapping Area: {oa:.4f} | Anomaly Score: {score:.4f}")

    patient_auc = roc_auc_score(np.array(patient_y_true) == -1, patient_scores)
    
    # --- Generate plots if plot_dir is provided ---
    if plot_dir is not None:
        import os
        from src.plotter_oa import (
            plot_single_patient_oa,
            plot_all_patients_overlay,
            plot_results_patients,
            make_split_plot,
        )
        
        if is_deep:
            metric_name = "Deep RBF Kernel"
        else:
            metric_name = f"kernel (m={mismatches}, k={max_k})"
        
        # 1. Per-patient OA plots (individual KDE curves + shading)
        per_patient_dir = os.path.join(plot_dir, "per_patient")
        for pdata in patient_plot_data:
            path = plot_single_patient_oa(
                xs, y_intra, pdata['y_inter'], pdata['oa'],
                patient_name=pdata['filename'],
                metric_name=metric_name,
                out_dir=per_patient_dir,
                seed=seed,
            )
            logger.info(f"  Saved per-patient plot: {path}")
        
        # 2. All patients overlay
        path = plot_all_patients_overlay(
            xs, y_intra, patient_plot_data,
            metric_name=metric_name,
            out_dir=plot_dir,
            seed=seed,
        )
        logger.info(f"  Saved all-patients overlay: {path}")
        
        # 3. Results bar chart
        path = plot_results_patients(
            patient_plot_data,
            metric_name=metric_name,
            out_dir=plot_dir,
            seed=seed,
        )
        logger.info(f"  Saved results bar chart: {path}")
        
        # 4. Split plot
        path = make_split_plot(
            patient_plot_data,
            metric_name=metric_name,
            out_dir=plot_dir,
            seed=seed,
        )
        logger.info(f"  Saved split plot: {path}")
    
    return patient_auc
