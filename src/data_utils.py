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


def load_clinical_data(
    normal_fasta: str, 
    tumor_fasta: str, 
    train_ratio: float = 0.8
) -> Tuple[List[str], List[str], np.ndarray]:
    """
    Loads real patient FASTA files. 
    Splits the Normal (Z) file into a train set and a healthy test set.
    Uses the Tumor (T) file as the anomalous test set.
    """
    print(f"Loading Matched Normal file: {normal_fasta}")
    normal_reader = MMapFastaReader(normal_fasta)
    normal_seqs = [normal_reader.get_seq(i) for i in range(len(normal_reader.offsets))]
    normal_reader.close()
    
    # Filter out any None values if the index had errors
    normal_seqs = [s for s in normal_seqs if s is not None]

    print(f"Loading Tumor file: {tumor_fasta}")
    tumor_reader = MMapFastaReader(tumor_fasta)
    tumor_seqs = [tumor_reader.get_seq(i) for i in range(len(tumor_reader.offsets))]
    tumor_reader.close()
    
    tumor_seqs = [s for s in tumor_seqs if s is not None]

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


def _vectorized_sequence_generator(
    num_sequences: int, 
    length_range: Tuple[int, int], 
    bases: List[str], 
    probs: List[float]
) -> List[str]:
    """
    Generates random DNA sequences efficiently using a vectorized 2D NumPy array.
    This minimizes the overhead of calling np.random repeatedly in a Python loop.
    
    Args:
        num_sequences: The number of sequences to generate.
        length_range: A tuple of (min_length, max_length).
        bases: List of characters representing the DNA alphabet.
        probs: List of probabilities corresponding to each base.
        
    Returns:
        A list of generated DNA sequences.
    """
    if num_sequences == 0:
        return []
        
    # 1. Determine the length of each individual sequence
    lengths = np.random.randint(length_range[0], length_range[1] + 1, size=num_sequences)
    max_len = np.max(lengths)
    
    # 2. Generate a massive 2D matrix of characters all at once (C-level speed)
    # Shape will be (num_sequences, max_len)
    char_matrix = np.random.choice(bases, size=(num_sequences, max_len), p=probs)
    
    # 3. Join the characters row by row, truncating at the specific random length 
    # assigned to that sequence in step 1.
    return ["".join(row[:l]) for row, l in zip(char_matrix, lengths)]

def generate_simulated_data(
    num_train: int = 4000,
    num_test_healthy: int = 950,
    num_test_cancer: int = 50,
    random_state: int = 42
) -> Tuple[List[str], List[str], np.ndarray]:
    """
    Generates synthetic DNA sequences simulating a strictly healthy baseline
    and a liquid biopsy test set containing both healthy and cancerous sequences.

    Args:
        num_train: Number of purely healthy sequences for the training set.
        num_test_healthy: Number of healthy sequences in the test set.
        num_test_cancer: Number of anomalous (cancer) sequences in the test set.
        random_state: Random seed for reproducibility.

    Returns:
        train_data: List of healthy DNA sequences.
        test_data: List of test DNA sequences (mixed).
        y_test_true: Array of ground truth labels (1 for normal, -1 for anomaly).
    """
    np.random.seed(random_state)
    bases = ['A', 'C', 'G', 'T', 'M']
    
    # Slight distribution shifts simulate the difference between healthy and mutated DNA
    healthy_probs = [0.28, 0.20, 0.22, 0.28, 0.02]
    cancer_probs  = [0.20, 0.20, 0.20, 0.20, 0.20]
    
    # Sequence length boundaries
    length_range = (120, 180)

    # Fast Vectorized Generation
    train_data = _vectorized_sequence_generator(num_train, length_range, bases, healthy_probs)
    test_healthy_data = _vectorized_sequence_generator(num_test_healthy, length_range, bases, healthy_probs)
    test_cancer_data = _vectorized_sequence_generator(num_test_cancer, length_range, bases, cancer_probs)

    # Combine testing data
    test_data = test_healthy_data + test_cancer_data
    
    # Generate ground truth labels (1 = Inlier/Healthy, -1 = Outlier/Cancer)
    # Matches the standard scikit-learn OneClassSVM anomaly output
    y_test_true = np.array([1] * num_test_healthy + [-1] * num_test_cancer)

    return train_data, test_data, y_test_true