"""
run_deep_mil_experiment.py
Executes a Deep Learning-Based Patient-Level Multiple Instance Learning (MIL) 
pipeline for Colon Cancer Novelty Detection.
"""

import time
import sys
import os
import numpy as np
from sklearn.metrics import roc_auc_score

# Dynamically resolve paths to ensure the script runs from anywhere
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Import from our custom library
from src.data_utils import MMapFastaReader
from src.DNAFeatureExtractor import compute_train_test_kernels
from src.evaluation import evaluate_novelty_detector
from experiments.experiments_utils import setup_logger, parse_arguments

logger = setup_logger(__name__)

def load_tracked_patient_cohort(train_normal_files, test_normal_files, test_tumor_files, args, logger):
    """
    Loads FASTA files while tracking the number of sequences extracted per file
    for Patient-Level MIL aggregation.
    """
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
    
    # Sequence-level ground truth (noisy)
    y_test_true_seq = np.array([1] * len(test_normal_data) + [-1] * len(test_tumor_data))
    
    return train_data, test_data, y_test_true_seq, test_files_info


def evaluate_patient_level_novelty(anomaly_scores, test_files_info, logger):
    """
    Aggregates sequence-level anomaly scores into a single Patient-Level score
    by taking the mean of the top 5% most anomalous sequences.
    """
    patient_y_true = []
    patient_scores = []
    current_idx = 0
    
    logger.info("\n--- Patient-Level Anomaly Aggregation ---")
    
    for info in test_files_info:
        num_seqs = info['num_sequences']
        
        # Extract sequences belonging ONLY to this patient
        seq_scores = anomaly_scores[current_idx : current_idx + num_seqs]
        current_idx += num_seqs
        
        # Scikit-learn OC-SVM: Lower scores are MORE anomalous. Invert them.
        inverted_scores = -seq_scores
        
        # Aggregate: Mean of the top 5% most anomalous reads
        top_k = max(1, int(num_seqs * 0.05)) 
        patient_score = np.mean(np.sort(inverted_scores)[-top_k:])
        
        patient_y_true.append(info['label'])
        patient_scores.append(patient_score)
        
        status = "TUMOR" if info['label'] == -1 else "HEALTHY"
        logger.info(f"[{status}] {info['filename']} -> Anomaly Score: {patient_score:.4f}")

    # Calculate Patient-Level ROC-AUC
    patient_auc = roc_auc_score(np.array(patient_y_true) == -1, patient_scores)
    return patient_auc


def main():
    args = parse_arguments(project_root)
    
    logger.info("=====================================================")
    logger.info(" COLON CANCER SOMATIC DETECTION: DEEP LEARNING MIL")
    logger.info("=====================================================")
    
    # --- 1. Define the Patient-Level Split ---
    train_normal_files = [
        os.path.join(args.data_dir, f"Healthy_{i}_merged_subset_1200000.fa") for i in range(2, 6)
    ]
    test_normal_files = [
        os.path.join(args.data_dir, f"Healthy_{i}_merged_subset_1200000.fa") for i in range(6, 8)
    ]
    test_tumor_files = [
        os.path.join(args.data_dir, f"Colo_{i}_merged_subset_1200000.fa") for i in range(1, 11) if i != 9
    ]
    
    # Verify files exist before running
    all_files = train_normal_files + test_normal_files + test_tumor_files
    missing_files = [f for f in all_files if not os.path.exists(f)]
    if missing_files:
        for f in missing_files:
            logger.error(f"Cannot find file: {f}")
        sys.exit(1)

    # --- 2. Load and Sample Tracked Data ---
    logger.info("Starting data loading and tracking...")
    train_data, test_data, y_test_true_seq, test_files_info = load_tracked_patient_cohort(
        train_normal_files,
        test_normal_files,
        test_tumor_files,
        args,
        logger
    )
    
    # --- 3. Deep Kernel Computation ---
    start_time = time.time()
    
    # K_train is (Train vs Train), K_test is (Test vs Train)
    K_train, K_test = compute_train_test_kernels(
        train_sequences=train_data,
        test_sequences=test_data,
        model_name="quietflamingo/dnabert2-no-flashattention", 
        kernel_type="rbf",
        batch_size=8
    )
    
    # --- 4. Sequence-Level Anomaly Detection ---
    logger.info(f"\nFitting One-Class SVM (nu={args.nu_param}) on Deep Kernels...")
    metrics = evaluate_novelty_detector(
        K_train=K_train, 
        K_test=K_test, 
        y_test_true=y_test_true_seq, 
        nu=args.nu_param
    )
    
    # --- 5. True Patient-Level Anomaly Aggregation ---
    patient_auc = evaluate_patient_level_novelty(
        anomaly_scores=metrics['anomaly_scores'], 
        test_files_info=test_files_info, 
        logger=logger
    )
    
    elapsed = time.time() - start_time
    
    # --- 6. Output Results ---
    logger.info("\n=====================================================")
    logger.info(" FINAL RESULTS: PATIENT-LEVEL MIL EVALUATION (DEEP)")
    logger.info("=====================================================")
    logger.info(f"Execution Time          : {elapsed:.2f} seconds")
    logger.info(f"Sequence-Level ROC-AUC  : {metrics['auc']:.4f} (Expected to be low/noisy)")
    logger.info(f"PATIENT-LEVEL ROC-AUC   : {patient_auc:.4f} (True Clinical Metric)")
    logger.info("=====================================================")

if __name__ == "__main__":
    main()