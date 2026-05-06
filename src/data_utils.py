"""
data_utils.py
Handles data simulation and generation for DNA sequence anomaly detection.
Optimized with vectorized NumPy operations for fast, large-scale dataset generation,
and Numba JIT compilation to eliminate I/O parsing bottlenecks.
"""

import os
import mmap
import logging
from array import array
from typing import Tuple, List

import numpy as np
import pysam
from numba import njit

# Configure the module-level logger
logger = logging.getLogger(__name__)

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
    Fast FASTA Reader based on mmap + .fai index + Numba JIT.
    Updated to support reading from read-only data directories via symlink caching.
    """

    def __init__(self, fasta_path: str, index_cache_dir: str = None):
        self.offsets = array('Q')
        self.lengths = array('I')

        # 1. Handle read-only directories by symlinking to a local cache
        if index_cache_dir:
            os.makedirs(index_cache_dir, exist_ok=True)
            filename = os.path.basename(fasta_path)
            symlink_path = os.path.join(index_cache_dir, filename)
            
            # Create a symlink to the original file if it doesn't exist
            if not os.path.exists(symlink_path):
                os.symlink(fasta_path, symlink_path)
            
            target_fasta_for_index = symlink_path
        else:
            target_fasta_for_index = fasta_path

        fai_path = target_fasta_for_index + '.fai'
        
        # 2. Generate the index if it doesn't exist (writes to the cache dir)
        if not os.path.exists(fai_path):
            logger.info(f"Cannot find .fai index, creating it: {fai_path}")
            pysam.faidx(target_fasta_for_index)

        # 3. Read the .fai data into the C arrays
        with open(fai_path, 'r') as f:
            for line in f:
                parts = line.rstrip('\n').split('\t')
                self.offsets.append(int(parts[2]))
                self.lengths.append(int(parts[1]))

        # 4. Open the FASTA file and create a memory map for zero-copy access
        self._file = open(fasta_path, 'rb')
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        
        # Expose the memory map to NumPy for Numba (zero-copy overhead)
        self._mmap_array = np.frombuffer(self._mmap, dtype=np.uint8)

    def get_seq(self, seq_idx: int) -> str | None:
        if seq_idx < 0 or seq_idx >= len(self.offsets):
            logger.warning(f"Index {seq_idx} out of range (max: {len(self.offsets) - 1})")
            return None

        offset = self.offsets[seq_idx]
        seq_len = self.lengths[seq_idx]

        # Call the optimized Numba function
        buf_array, read = _extract_sequence_fast(self._mmap_array, offset, seq_len)

        # Check if we read the expected number of characters (sanity check for corrupted .fai)
        if read < seq_len:
            logger.error(
                f"Corrupted sequence at index {seq_idx}: "
                f"expected {seq_len} chars, got only {read}. "
                f"Regenerate .fai with: samtools faidx <file>"
            )
            return None

        # Convert the numpy array back to bytes and decode
        return buf_array.tobytes().decode('ascii')

    def close(self):
        """Close the memory map and file handle."""
        # 1. Delete the NumPy array reference to release the exported memory pointer
        if hasattr(self, '_mmap_array'):
            del self._mmap_array
            
        # 2. Now it is safe to close the memory map and the file
        self._mmap.close()
        self._file.close()


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