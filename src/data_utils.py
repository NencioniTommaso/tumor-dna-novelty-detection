"""
data_utils.py
Handles data simulation and generation for DNA sequence anomaly detection.
Optimized with vectorized NumPy operations for fast, large-scale dataset generation.
"""

import numpy as np
from typing import Tuple, List

import os
import mmap
from array import array
import pysam

class MMapFastaReader:
    """
    Reader FASTA ultra-veloce basato su mmap + .fai index.
    L'indice è memorizzato in due array di tipo C per minimizzare il footprint.
    """

    def __init__(self, fasta_path: str):
        fai_path = fasta_path + '.fai'
        if not os.path.exists(fai_path):
            print(f"Indice .fai non trovato, lo creo: {fai_path}")
            pysam.faidx(fasta_path)

        # Array C per offset (unsigned long long, 8 byte) e length (unsigned int, 4 byte)
        self.offsets = array('Q')
        self.lengths = array('I')

        # Popola gli array leggendo il .fai
        with open(fai_path, 'r') as f:
            for line in f:
                parts = line.rstrip('\n').split('\t')
                # parts[2]: offset, parts[1]: lunghezza
                self.offsets.append(int(parts[2]))
                self.lengths.append(int(parts[1]))

        # Apri il FASTA in binario e crea la memory‐map
        self._file = open(fasta_path, 'rb')
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

    def get_seq(self, seq_idx: int) -> str | None:
        if seq_idx < 0 or seq_idx >= len(self.offsets):
            import logging
            logging.warning(f"Index {seq_idx} out of range (max: {len(self.offsets) - 1})")
            return None

        offset = self.offsets[seq_idx]
        seq_len = self.lengths[seq_idx]

        pos = offset
        read = 0
        buf = bytearray(seq_len)
        while read < seq_len and pos < len(self._mmap):
            c = self._mmap[pos]
            if c != 10:
                buf[read] = c
                read += 1
            pos += 1

        # CONTROLLO: hai letto tutti i caratteri?
        if read < seq_len:
            import logging
            logging.error(
                f"Corrupted sequence at index {seq_idx}: "
                f"expected {seq_len} chars, got only {read}. "
                f"Regenerate .fai with: samtools faidx <file>"
            )
            return None

        return buf.decode('ascii')

    def close(self):
        """Chiude la mappa e il file handle."""
        self._mmap.close()
        self._file.close()
    
    pass


import re

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


def load_clinical_data(
    normal_fasta: str, 
    tumor_fasta: str, 
    train_ratio: float = 0.8
) -> Tuple[List[str], List[str], np.ndarray]:
    """
    Loads real patient FASTA files. 
    Splits the Normal (Z) file into a train set and a healthy test set.
    Uses the Tumor (T) file as the anomalous test set.
    Automatically cleans sequences of IUPAC ambiguity codes.
    """
    print(f"Loading Matched Normal file: {normal_fasta}")
    normal_reader = MMapFastaReader(normal_fasta)
    normal_seqs_raw = [normal_reader.get_seq(i) for i in range(len(normal_reader.offsets))]
    normal_reader.close()
    
    # Filter out any None values AND clean the valid sequences
    normal_seqs = [clean_sequence(s) for s in normal_seqs_raw if s is not None]

    print(f"Loading Tumor file: {tumor_fasta}")
    tumor_reader = MMapFastaReader(tumor_fasta)
    tumor_seqs_raw = [tumor_reader.get_seq(i) for i in range(len(tumor_reader.offsets))]
    tumor_reader.close()
    
    # Filter out any None values AND clean the valid sequences
    tumor_seqs = [clean_sequence(s) for s in tumor_seqs_raw if s is not None]

    # --- Train/Test Split ---
    # We use e.g., 80% of the healthy data to build the baseline model
    num_train = int(len(normal_seqs) * train_ratio)
    
    train_data = normal_seqs[:num_train]
    test_healthy_data = normal_seqs[num_train:]
    test_cancer_data = tumor_seqs
    
    test_data = test_healthy_data + test_cancer_data
    
    # Ground truth labels for evaluation: 1 for normal, -1 for tumor
    y_test_true = np.array([1] * len(test_healthy_data) + [-1] * len(test_cancer_data))
    
    print(f"Loaded {len(train_data)} Train (Normal), {len(test_healthy_data)} Test (Normal), {len(test_cancer_data)} Test (Tumor)")
    
    return train_data, test_data, y_test_true