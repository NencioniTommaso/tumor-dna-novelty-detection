# Data Loading and FASTA I/O

> **File**: `src/data_utils.py`
> This module handles reading DNA sequences from FASTA files at high speed using memory-mapped I/O, `.fai` indexing, and Numba JIT compilation.

## FASTA File Format

FASTA is the standard format for biological sequence data. Each entry has a header line starting with `>` followed by the sequence on subsequent lines:

```
>sequence_0001
ACGTACGTACGTACGTACGTACGTACGT
ACGTACGTACGTACGTACGT
>sequence_0002
TGCATGCATGCATGCATGCATGCATGCA
```

Our dataset files follow the naming convention:
- `Healthy_N_merged_subset_1200000.fa` — Healthy patient N (cfDNA from a cancer-free individual)
- `Colo_N_merged_subset_1200000.fa` — Colon cancer patient N (cfDNA containing circulating tumor DNA)

Each file contains ~1,200,000 sequences.

---

## MMapFastaReader: Zero-Copy Sequence Access

### The Problem

Reading 1.2 million sequences from disk using standard Python I/O is slow. The sequences need to be randomly sampled (not read sequentially), making line-by-line parsing even worse.

### The Solution

`MMapFastaReader` uses three optimizations:

#### 1. `.fai` Indexing (pysam)

A `.fai` index file stores the byte offset and length of every sequence in the FASTA file. This enables **random access** — jumping directly to any sequence by its index without scanning through the file.

```
.fai format:
seq_name    seq_length    byte_offset    line_bases    line_width
seq_0001    150           17              80            81
seq_0002    150           198             80            81
```

The index is generated once (by `pysam.faidx`) and cached.

#### 2. Memory-Mapped I/O (mmap)

Instead of reading the file into a Python buffer, `mmap` maps the file directly into the process's virtual address space. The operating system handles caching and paging transparently. This avoids:
- Copying data from kernel space to user space.
- Allocating Python string objects for each sequence.

```python
self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
self._mmap_array = np.frombuffer(self._mmap, dtype=np.uint8)
```

The `np.frombuffer` call creates a NumPy view over the mmap — **zero copies**.

#### 3. Numba JIT Compilation

The sequence extraction function (`_extract_sequence_fast`) is compiled to machine code by Numba:

```python
@njit
def _extract_sequence_fast(mmap_array, offset, seq_len):
    buf = np.empty(seq_len, dtype=np.uint8)
    pos = offset
    read = 0
    while read < seq_len and pos < len(mmap_array):
        c = mmap_array[pos]
        if c != 10:  # Skip newlines
            buf[read] = c
            read += 1
        pos += 1
    return buf, read
```

This runs at C speed, bypassing the Python interpreter entirely. The function filters out newline characters (ASCII 10) that break up sequences across lines in the FASTA format.

### Read-Only Directory Support

The data directory (`/read_only/dir`) may be read-only (e.g., a shared NFS mount). The `.fai` index needs to be written *somewhere*. The solution:

1. Create a symlink in a writable cache directory pointing to the original FASTA file.
2. Generate the `.fai` index next to the symlink (in the cache directory).
3. On subsequent runs, reuse the cached index.

```
/read_only/dir/Healthy_2.fa                ← symlink → original
data/.fai_cache/Healthy_2.fa                ← symlink → original
data/.fai_cache/Healthy_2.fa.fai            ← writable index
```

---

## Cohort Loading

### `load_tracked_patient_cohort`

This function orchestrates loading the full experimental cohort:

```python
train_data, test_data, y_test_true_seq, test_files_info = load_tracked_patient_cohort(
    train_normal_files,    # Healthy files for training
    test_normal_files,     # Healthy files for testing
    test_tumor_files,      # Tumor files for testing
    max_train,             # Total training sequences to sample
    max_test_normal,       # Total healthy test sequences
    max_test_tumor,        # Total tumor test sequences
    seed,                  # Random seed for reproducibility
    cache_dir,             # Path for .fai index cache
    logger
)
```

### Sequence Sampling

Sequences are sampled **uniformly at random** from each file using `np.random.choice`:

```python
seqs_per_file = max_total_seqs // len(file_list)
sampled_indices = np.random.choice(total_available, num_to_sample, replace=False)
```

The total budget is split evenly across files. With `max_train = 25000` and 4 training files, each file contributes ~6,250 sequences.

### The `test_files_info` Tracker

For patient-level aggregation, we need to know which sequences came from which patient. The tracker records:

```python
{
    'filename': 'Colo_3_merged_subset_1200000.fa',
    'label': -1,          # -1 = tumor, 1 = healthy
    'num_sequences': 1000  # How many sequences were sampled from this patient
}
```

This allows `evaluate_patient_level_novelty` to slice the flat anomaly score array back into per-patient groups.

### Deterministic Reproducibility

The function calls `np.random.seed(seed)` at the top. Because sampling is done sequentially (file by file, training then testing), the exact same sequences are selected every time given the same seed and parameters. This is critical for comparing results across runs.

> **Warning**: If you change `max_train`, the random seed advances a different number of times during training sampling, which changes *all* subsequent test samples too. This is by design — the seed state is sequential.

---

## Patient Cohort Split

The dataset is split at the **patient level**, not the sequence level:

| Split | Patients | Purpose |
|---|---|---|
| Training | Healthy 2, 3, 4, 5 | Learn what "normal" looks like |
| Testing (Healthy) | Healthy 6, 7 | Verify the model doesn't flag healthy patients |
| Testing (Tumor) | Colo 1–8, 10 | Verify the model detects tumor patients |
| Validation | Same as Testing | Used by `calibrate_threshold.py` for threshold tuning |
