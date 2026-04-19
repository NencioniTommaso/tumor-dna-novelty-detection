"""
data_utils.py
Handles data simulation and generation for DNA sequence anomaly detection.
Optimized with vectorized NumPy operations for fast, large-scale dataset generation,
and Numba JIT compilation to eliminate I/O parsing bottlenecks.
"""

import numpy as np
from typing import Tuple, List

import os
import mmap
from array import array
import pysam
import re

from numba import njit

# --- NUMBA OPTIMIZATION ---
# @njit forces this function to compile to raw machine code. 
# It loops through the memory map array and filters out newlines (ASCII 10) 
# at C-like speeds, bypassing the Python Global Interpreter Lock (GIL).
@njit
def _extract_sequence_fast(mmap_array: np.ndarray, offset: int, seq_len: int):
    # Allocate a flat array to hold the clean bytes
    buf = np.empty(seq_len, dtype=np.uint8)
    pos = offset
    read = 0
    max_len = len(mmap_array)
    
    while read < seq_len and pos < max_len:
        c = mmap_array[pos]
        if c != 10:  # Ignore '\n'
            buf[read] = c
            read += 1
        pos += 1
        
    return buf, read

class MMapFastaReader:
    """
    Reader FASTA ultra-veloce basato su mmap + .fai index + Numba JIT.
    L'indice è memorizzato in due array di tipo C per minimizzare il footprint.
    """

    def __init__(self, fasta_path: str):
        fai_path = fasta_path + '.fai'
        if not os.path.exists(fai_path):
            print(f"Indice .fai non trovato, lo creo: {fai_path}")
            pysam.faidx(fasta_path)

        # C Array (unsigned long long, 8 byte), length (unsigned int, 4 byte)
        self.offsets = array('Q')
        self.lengths = array('I')

        # write the .fai data into the C arrays for fast access
        with open(fai_path, 'r') as f:
            for line in f:
                parts = line.rstrip('\n').split('\t')
                # parts[2]: offset, parts[1]: lunghezza
                self.offsets.append(int(parts[2]))
                self.lengths.append(int(parts[1]))

        # Open the FASTA file and create a memory map for zero-copy access
        self._file = open(fasta_path, 'rb')
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        
        # Expose the memory map to NumPy for Numba (zero-copy overhead)
        self._mmap_array = np.frombuffer(self._mmap, dtype=np.uint8)

    def get_seq(self, seq_idx: int) -> str | None:
        if seq_idx < 0 or seq_idx >= len(self.offsets):
            import logging
            logging.warning(f"Index {seq_idx} out of range (max: {len(self.offsets) - 1})")
            return None

        offset = self.offsets[seq_idx]
        seq_len = self.lengths[seq_idx]

        # Call the optimized Numba function
        buf_array, read = _extract_sequence_fast(self._mmap_array, offset, seq_len)

        # Check if we read the expected number of characters (sanity check for corrupted .fai)
        if read < seq_len:
            import logging
            logging.error(
                f"Corrupted sequence at index {seq_idx}: "
                f"expected {seq_len} chars, got only {read}. "
                f"Regenerate .fai with: samtools faidx <file>"
            )
            return None

        # Convert the numpy array back to bytes and decode
        return buf_array.tobytes().decode('ascii')

    def close(self):
        """Chiude la mappa e il file handle."""
        # 1. Delete the NumPy array reference to release the exported memory pointer
        if hasattr(self, '_mmap_array'):
            del self._mmap_array
            
        # 2. Now it is safe to close the memory map and the file
        self._mmap.close()
        self._file.close()


def load_patient_cohort(
    train_normal_files: List[str], 
    test_normal_files: List[str], 
    test_tumor_files: List[str],
    max_train: int = 5000,
    max_test_normal: int = 2500,
    max_test_tumor: int = 2500,
    random_seed: int = 42
) -> Tuple[List[str], List[str], np.ndarray]:
    """
    Loads patient FASTA files and creates the split.
    Uses index-level random sampling to efficiently create balanced train/test sets without loading entire files into memory.
    """
    np.random.seed(random_seed)
    
    def _read_sampled_files(file_list: List[str], desc: str, max_total_seqs: int) -> List[str]:
        if not file_list:
            return []
            
        seqs_per_file = max_total_seqs // len(file_list)
        all_seqs = []
        
        for file_path in file_list:
            print(f"  -> Loading {desc}: {os.path.basename(file_path)} (Target: {seqs_per_file} seqs)")
            reader = MMapFastaReader(file_path)
            total_available = len(reader.offsets)
            
            # Determine how many sequences we can safely sample
            num_to_sample = min(seqs_per_file, total_available)
            
            # Fast index sampling: pick random indices without replacement
            sampled_indices = np.random.choice(total_available, num_to_sample, replace=False)
            
            # Extract ONLY the sampled sequences
            raw_seqs = [reader.get_seq(i) for i in sampled_indices]
            reader.close()
            
            # Clean sequences and drop any Nones
            clean_seqs = [clean_sequence(s) for s in raw_seqs if s is not None]
            all_seqs.extend(clean_seqs)
            
        return all_seqs

    print("\n--- Loading Training Data (Healthy Baseline) ---")
    train_data = _read_sampled_files(train_normal_files, "Train (Normal)", max_train)
    
    print("\n--- Loading Testing Data (Inliers & Outliers) ---")
    test_healthy_data = _read_sampled_files(test_normal_files, "Test (Normal)", max_test_normal)
    test_cancer_data = _read_sampled_files(test_tumor_files, "Test (Tumor)", max_test_tumor)
    
    test_data = test_healthy_data + test_cancer_data
    
    # Ground truth labels: 1 for normal, -1 for tumor
    y_test_true = np.array([1] * len(test_healthy_data) + [-1] * len(test_cancer_data))
    
    print(f"\n[Data Load Complete]")
    print(f"Train (Normal): {len(train_data)} sequences")
    print(f"Test  (Normal): {len(test_healthy_data)} sequences")
    print(f"Test  (Tumor) : {len(test_cancer_data)} sequences")
    print(f"Total Combined: {len(train_data) + len(test_data)} sequences")
    
    return train_data, test_data, y_test_true

def clean_sequence(sequence):
    """
    Replaces non-standard IUPAC ambiguity codes (like M, H, K, etc.) with 'N'.
    This prevents the string kernel's feature vocabulary from artificially exploding.
    """
    return re.sub(r'[^ACGT]', 'N', sequence.upper())


def chunk_sequence(sequence, chunk_size=200):
    """
    Breaks a sequence into chunks. 
    Handles variable-length reads: if a sequence is shorter than chunk_size, 
    it is kept as a single smaller chunk.
    """
    seq_clean = clean_sequence(sequence)
    
    if len(seq_clean) <= chunk_size:
        return [seq_clean]
        
    return [seq_clean[i:i+chunk_size] for i in range(0, len(seq_clean), chunk_size)]
