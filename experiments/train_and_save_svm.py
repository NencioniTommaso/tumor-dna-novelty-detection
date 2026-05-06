"""
train_and_save_svm.py
Executes the MIL pipeline, evaluates patient-level ROC-AUC, 
and saves the trained precomputed kernel SVM for future inference.
"""

import time
import sys
import os
import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.data_utils import MMapFastaReader
from src.kernels import mixed_string_kernel, normalize_gram
from experiments.experiments_utils import setup_logger, parse_arguments, generate_mkl_weights

logger = setup_logger(__name__)

def load_tracked_patient_cohort(train_normal_files, test_normal_files, test_tumor_files, args, logger):
    np.random.seed(args.seed)
    
    def _read_and_track(file_list, desc, max_total_seqs, label):
        if not file_list:
            return [], []
        seqs_per_file = max_total_seqs // len(file_list)
        all_seqs = []
        files_info = []
        
        for file_path in file_list:
            logger.info(f"  -> Loading {desc}: {os.path.basename(file_path)}")
            reader = MMapFastaReader(file_path, index_cache_dir=args.cache_dir)
            total_available = len(reader.offsets)
            num_to_sample = min(seqs_per_file, total_available)
            sampled_indices = np.random.choice(total_available, num_to_sample, replace=False)
            raw_seqs = [reader.get_seq(i) for i in sampled_indices]
            reader.close()
            
            clean_seqs = [s.upper() for s in raw_seqs if s is not None]
            all_seqs.extend(clean_seqs)
            
            if label is not None:
                files_info.append({
                    'filename': os.path.basename(file_path),
                    'label': label,
                    'num_sequences': len(clean_seqs)
                })
        return all_seqs, files_info

    logger.info("--- Loading Training Data (Healthy Baseline) ---")
    train_data, _ = _read_and_track(train_normal_files, "Train (Normal)", args.max_train, None)
    
    logger.info("\n--- Loading Testing Data (Tracked Instances) ---")
    test_normal_data, normal_info = _read_and_track(test_normal_files, "Test (Normal)", args.max_test_normal, 1)
    test_tumor_data, tumor_info = _read_and_track(test_tumor_files, "Test (Tumor)", args.max_test_tumor, -1)
    
    test_data = test_normal_data + test_tumor_data
    test_files_info = normal_info + tumor_info
    y_test_true_seq = np.array([1] * len(test_normal_data) + [-1] * len(test_tumor_data))
    
    return train_data, test_data, y_test_true_seq, test_files_info

def evaluate_patient_level_novelty(anomaly_scores, test_files_info, logger):
    patient_y_true, patient_scores = [], []
    current_idx = 0
    logger.info("\n--- Patient-Level Anomaly Aggregation ---")
    for info in test_files_info:
        num_seqs = info['num_sequences']
        seq_scores = anomaly_scores[current_idx : current_idx + num_seqs]
        current_idx += num_seqs
        
        inverted_scores = -seq_scores
        top_k = max(1, int(num_seqs * 0.05)) 
        patient_score = np.mean(np.sort(inverted_scores)[-top_k:])
        
        patient_y_true.append(info['label'])
        patient_scores.append(patient_score)
        
        status = "TUMOR" if info['label'] == -1 else "HEALTHY"
        logger.info(f"[{status}] {info['filename']} -> Anomaly Score: {patient_score:.4f}")

    patient_auc = roc_auc_score(np.array(patient_y_true) == -1, patient_scores)
    return patient_auc

def main():
    args = parse_arguments(project_root)
    logger.info("=====================================================")
    logger.info(" COLON CANCER SOMATIC DETECTION: MIL & MODEL SAVING")
    logger.info("=====================================================")
    
    train_normal_files = [os.path.join(args.data_dir, f"Healthy_{i}_merged_subset_1200000.fa") for i in range(2, 6)]
    test_normal_files = [os.path.join(args.data_dir, f"Healthy_{i}_merged_subset_1200000.fa") for i in range(6, 8)]
    test_tumor_files = [os.path.join(args.data_dir, f"Colo_{i}_merged_subset_1200000.fa") for i in range(1, 11) if i != 9]
    
    train_data, test_data, y_test_true_seq, test_files_info = load_tracked_patient_cohort(
        train_normal_files, test_normal_files, test_tumor_files, args, logger
    )
    
    mkl_weights = generate_mkl_weights(args.max_k, args.mismatches)
    logger.info(f"\nComputing Explicit Sparse Mismatch Kernel...")
    
    K_full, _ = mixed_string_kernel(
        sequences=train_data + test_data, k_max=args.max_k, m=args.mismatches, 
        weights=mkl_weights, n_jobs=args.n_jobs  
    )
    K_full = normalize_gram(K_full)
    
    num_train = len(train_data)
    K_train = K_full[:num_train, :num_train]
    K_test = K_full[num_train:, :num_train]
    
    logger.info(f"\nFitting One-Class SVM (nu={args.nu_param})...")
    
    # We construct the SVM explicitly here so we can save it
    from sklearn.svm import OneClassSVM
    svm = OneClassSVM(kernel='precomputed', nu=args.nu_param)
    svm.fit(K_train)
    
    anomaly_scores = svm.decision_function(K_test)
    patient_auc = evaluate_patient_level_novelty(anomaly_scores, test_files_info, logger)
    
    logger.info(f"\nPATIENT-LEVEL ROC-AUC : {patient_auc:.4f}")
    
    # --- SAVE THE ARTIFACTS ---
    save_dir = os.path.join(project_root, "models")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "ocsvm_pretrained.pkl")
    
    logger.info(f"\nSaving pre-trained SVM and {len(train_data)} training sequences to {save_path}...")
    saved_state = {
        'model': svm,
        'train_sequences': train_data, 
        'nu': args.nu_param,
        'max_k': args.max_k,
        'mismatches': args.mismatches
    }
    joblib.dump(saved_state, save_path)
    logger.info("Model saved successfully! Ready for inference.")

if __name__ == "__main__":
    main()