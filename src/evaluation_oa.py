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
    
    return info['label'], patient_score, oa, info['filename']

def compute_distances(K: np.ndarray) -> np.ndarray:
    """
    Computes distances given a normalized kernel matrix (where K(x,x)=1).
    D^2(x,y) = K(x,x) + K(y,y) - 2K(x,y) = 2 - 2K(x,y)
    """
    dist_sq = 2.0 * (1.0 - K)
    dist_sq = np.clip(dist_sq, a_min=0, a_max=None)
    return np.sqrt(dist_sq)

def compute_kde(distances: np.ndarray, xmax: float = None, num_points: int = 512, downsample: bool = True, max_samples: int = 100000):
    """
    Fits a Gaussian KDE to the distances and evaluates it on a grid.
    Mirrors Innocenti's kde.py.
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
    Mirrors Innocenti's overllappingArea.py.
    """
    area = np.trapezoid(np.abs(y_intra - y_inter), xs)
    OA = 1.0 - area / 2.0
    return OA

def evaluate_patient_level_oa_method(K_train, K_test, test_files_info, logger, n_jobs=-1, downsample_kde=True):
    """
    Computes Overlapping Area (OA) per patient exactly as Innocenti does.
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
    
    for label, score, oa, filename in results:
        patient_y_true.append(label)
        patient_scores.append(score)
        
        status = "TUMOR" if label == -1 else "HEALTHY"
        logger.info(f"[{status}] {filename} -> Overlapping Area: {oa:.4f} | Anomaly Score: {score:.4f}")

    patient_auc = roc_auc_score(np.array(patient_y_true) == -1, patient_scores)
    return patient_auc
