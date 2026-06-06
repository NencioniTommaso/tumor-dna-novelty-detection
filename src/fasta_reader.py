"""
fasta_reader.py
Fast FASTA file reading using mmap, .fai indexing, and Numba JIT compilation.
Provides the MMapFastaReader class as a self-contained, reusable I/O primitive.
"""

import os
import mmap
import logging
from array import array

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
        fasta_path = os.path.abspath(fasta_path)
        if not os.path.exists(fasta_path):
            raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

        # 1. Handle read-only directories by symlinking to a local cache
        if index_cache_dir:
            os.makedirs(index_cache_dir, exist_ok=True)
            filename = os.path.basename(fasta_path)
            symlink_path = os.path.join(index_cache_dir, filename)
            
            # Reuse an existing valid cache link; otherwise replace stale entries.
            if os.path.lexists(symlink_path):
                if os.path.islink(symlink_path) and os.readlink(symlink_path) == fasta_path:
                    pass
                else:
                    os.remove(symlink_path)
                    os.symlink(fasta_path, symlink_path)
            else:
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
