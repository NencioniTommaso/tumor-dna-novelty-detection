"""
run_inference.py
Loads a pre-trained Patient-Level MIL model and efficiently calculates 
only the asymmetric inference kernel for new patients, avoiding memory bloat.
"""

import os
import sys
import joblib
import numpy as np
import argparse

# Dynamically resolve paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Import the optimized asymmetric kernel from our core library
from src.kernels import compute_asymmetric_normalized_kernel
from src.data_utils import MMapFastaReader
from experiments.experiments_utils import generate_mkl_weights

def run_saved_model_inference(patient_fasta_path, model_path, cache_dir, sample_size=1500):
    """
    Loads a pretrained SVM and scores a single new patient FASTA file.
    """
    print(f"Loading pretrained model from {model_path}...")
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        sys.exit(1)
        
    saved_state = joblib.load(model_path)
    svm = saved_state['model']
    train_sequences = saved_state['train_sequences']
    max_k = saved_state['max_k']
    mismatches = saved_state['mismatches']
    
    print(f"Loaded SVM trained on {len(train_sequences)} sequences.")
    
    # --- Load Patient Data ---
    print(f"Loading patient data from {os.path.basename(patient_fasta_path)}...")
    reader = MMapFastaReader(patient_fasta_path, index_cache_dir=cache_dir)
    total_available = len(reader.offsets)
    num_to_sample = min(sample_size, total_available)
    
    # Fast index sampling
    sampled_indices = np.random.choice(total_available, num_to_sample, replace=False)
    raw_seqs = [reader.get_seq(i) for i in sampled_indices]
    reader.close()
    
    new_patient_sequences = [s.upper() for s in raw_seqs if s is not None]
    
    # --- Compute Optimized Inference Kernel ---
    print(f"Computing asymmetric inference kernel for {len(new_patient_sequences)} sequences...")
    mkl_weights = generate_mkl_weights(max_k, mismatches)
    K_test = compute_asymmetric_normalized_kernel(
        test_seqs=new_patient_sequences, 
        train_seqs=train_sequences, 
        max_k=max_k, 
        mismatches=mismatches, 
        mkl_weights=mkl_weights
    )
    
    # --- Run Sequence-Level Inference ---
    print("Predicting sequence anomalies...")
    anomaly_scores = svm.decision_function(K_test)
    
    # --- Aggregate to Patient Level (MIL) ---
    inverted_scores = -anomaly_scores
    top_k = max(1, int(len(new_patient_sequences) * 0.05)) # Top 5% most anomalous
    patient_score = np.mean(np.sort(inverted_scores)[-top_k:])
    
    print("\n=====================================================")
    print(f" PATIENT FASTA: {os.path.basename(patient_fasta_path)}")
    print(f" FINAL ANOMALY SCORE: {patient_score:.4f}")
    print("=====================================================\n")
    
    return patient_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference using a saved OC-SVM model.")
    parser.add_argument("--patient-file", required=True, help="Path to the patient's FASTA file.")
    parser.add_argument("--model-path", default=os.path.join(project_root, "models", "ocsvm_pretrained.pkl"), help="Path to the saved .pkl model.")
    parser.add_argument("--sample-size", type=int, default=1500, help="Number of sequences to sample from the patient.")
    
    args = parser.parse_args()
    cache_dir = os.path.join(project_root, ".fai_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    run_saved_model_inference(args.patient_file, args.model_path, cache_dir, args.sample_size)