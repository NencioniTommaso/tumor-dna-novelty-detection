# String Kernels and Feature Extraction

> **File**: `src/kernels.py`
> This is the computational heart of the project. It transforms raw DNA sequences into a mathematical space where similarity can be measured, then computes the kernel (Gram) matrices that the SVM operates on.

## The Core Idea

Traditional machine learning needs numerical feature vectors. DNA sequences are strings of characters. **String kernels** bridge this gap: they define a similarity function between two sequences *without* ever explicitly computing a finite-dimensional feature vector (though in practice we do compute sparse feature vectors for efficiency).

The specific kernel used here is a variant of the **(k, m)-Mismatch String Kernel**:
- Extract all **k-mers** (substrings of length k) from each sequence.
- For each observed k-mer, also count its **mismatch neighborhood** — all k-mers within Hamming distance *m*.
- The kernel value between two sequences is the dot product of their mismatch-expanded k-mer frequency vectors.

This is biologically meaningful: a somatic mutation (e.g., a C→T point mutation) will change a small number of k-mers in the tumor sequence. The mismatch tolerance ensures that near-matches still contribute to similarity, making the kernel robust to sequencing errors while still being sensitive to real mutations.

---

## The Epigenetic Alphabet

```python
EPIGENETIC_ALPHABET = ('A', 'C', 'G', 'T', 'M', 'H')
```

Beyond the standard 4 DNA bases, the system supports two additional characters:
- **M**: Methylated cytosine (5-methylcytosine)
- **H**: Hydroxymethylated cytosine (5-hydroxymethylcytosine)

These epigenetic modifications are common biomarkers in cancer. By including them in the alphabet, the mismatch kernel can distinguish between a normal cytosine and a methylated one, treating them as distinct biological signals rather than conflating them.

---

## Mismatch Neighborhood Generation

```
Example: kmer = "ACG", m = 1, alphabet = {A, C, G, T, M, H}

The neighborhood includes "ACG" itself plus every k-mer
that differs in exactly 1 position:

Position 0 mutated: CCG, GCG, TCG, MCG, HCG
Position 1 mutated: AAG, AGG, ATG, AMG, AHG
Position 2 mutated: ACA, ACC, ACT, ACM, ACH

Total: 1 (original) + 15 (mutants) = 16 neighbors
```

The function `generate_mismatch_neighborhood(kmer, m)` computes this set. It is cached with `@lru_cache` because the same k-mer appears many times across sequences.

### Weighted Mismatch Variant

The function `generate_weighted_mismatch_neighborhood` extends this by tracking the **Hamming distance** of each neighbor from the original k-mer. This enables **distance-weighted counting**:

```
weight = mismatch_decay ^ hamming_distance
```

With `mismatch_decay = 0.5`:
- Exact match: weight = 0.5⁰ = 1.0
- 1-mismatch neighbor: weight = 0.5¹ = 0.5

This makes inexact matches contribute less to the kernel, providing a smoother similarity measure than the binary mismatch kernel.

---

## Feature Extraction Pipeline

### Step 1: Vocabulary Construction

For each k-mer length k, the system pre-enumerates **all possible k-mers** from the 6-letter alphabet:

```
vocabulary size = |alphabet|^k = 6^k
```

| k | Vocabulary Size |
|---|---|
| 3 | 216 |
| 4 | 1,296 |
| 5 | 7,776 |
| 6 | 46,656 |

This fixed vocabulary enables fully independent parallel processing — each worker can map k-mers to column indices without synchronization.

### Step 2: Sparse Feature Matrix

For each sequence, we:
1. Extract all k-mers by sliding a window of size k.
2. Expand each k-mer to its (weighted) mismatch neighborhood.
3. Accumulate counts into a row of a sparse matrix.

The result is a sparse matrix **X** of shape `(n_sequences, vocabulary_size)` in CSR format. Sparsity is critical: a sequence of length 150 with k=6 has only 145 k-mers, but the vocabulary has 46,656 entries. Most entries are zero.

### Step 3: Gram Matrix Computation

The kernel (Gram) matrix is:

```
K[i, j] = X[i] · X[j]^T
```

This is a single sparse matrix multiplication: `K = X @ X.T`

---

## Multiple Kernel Learning (MKL)

Rather than choosing a single k, we compute kernels at **multiple k-mer lengths** and combine them:

```
K_final = w₃·K₃ + w₄·K₄ + w₅·K₅ + w₆·K₆
```

### Weight Generation (`generate_mkl_weights`)

Not all k-mer lengths are equally informative:
- **Short k-mers (k ≤ 2)**: Dominated by base composition noise. Suppressed to weight 0.
- **Longer k-mers**: Carry more specific structural signal. Weighted linearly by `k - noise_threshold`.

The `noise_threshold` defaults to `max(1, 2 * mismatches)`. With `mismatches = 1`, this means k=1 and k=2 are suppressed.

Example with `max_k = 6, mismatches = 1`:
```
Raw weights: [0, 0, 1, 2, 3, 4]  (k=1,2 suppressed)
Normalized:  [0, 0, 0.1, 0.2, 0.3, 0.4]
```

The weight is applied by scaling the feature matrix: `X_k *= sqrt(w_k)`, so that when the dot product is taken, the contribution is `w_k * (X_k · X_k^T)`.

---

## Parallelization Strategy

The kernel computation is the most expensive part of the pipeline. The code uses a two-level parallelization strategy:

### Level 1: Sequential Over k-mer Lengths

Each k-mer length is processed sequentially. This avoids memory contention — each sub-kernel can be large.

### Level 2: Parallel Within Each k

Within a single k, the Gram matrix is split into **blocks** and computed in parallel using `joblib`:

```
┌────┬────┬────┐
│ B₁ │ B₂ │ B₃ │  ← Row blocks
├────┼────┼────┤
│ B₂ᵀ│ B₄ │ B₅ │  ← Only upper triangle computed
├────┼────┼────┤     (symmetric matrix)
│ B₃ᵀ│ B₅ᵀ│ B₆ │
└────┴────┴────┘
```

For the symmetric training kernel, only the upper triangle is computed and mirrored. The block size defaults to 1500 sequences.

### Thread Pinning

```python
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
```

BLAS libraries (used internally by NumPy/SciPy) spawn their own threads. When combined with joblib's process-level parallelism, this causes thread contention. Pinning BLAS to single-threaded mode forces all parallelism through joblib.

---

## Gram Matrix Normalization

After computing the raw Gram matrix, it is normalized:

```
K_norm[i, j] = K[i, j] / sqrt(K[i, i] * K[j, j])
```

This is equivalent to computing the **cosine similarity** in the feature space. It ensures that:
- `K_norm[i, i] = 1` for all sequences.
- Similarity is not dominated by sequence length differences.

---

## Asymmetric (Inference) Kernel

During training, we compute `K_train = X_train @ X_train.T` (symmetric, square).

During inference, we need `K_test = X_test @ X_train.T` (rectangular). This is the **asymmetric kernel** — it measures the similarity of each test sequence against each training sequence.

### Normalization of the Asymmetric Kernel

The asymmetric kernel requires the diagonal self-similarity of *both* test and train sequences:

```
K_test_norm[i, j] = K_test[i, j] / sqrt(diag_test[i] * diag_train[j])
```

Where:
- `diag_test[i] = X_test[i] · X_test[i]^T` (test self-similarity)
- `diag_train[j] = X_train[j] · X_train[j]^T` (training self-similarity, saved from training)

The training diagonal (`diag_train`) is stored in `train_states` during training, so it does not need to be recomputed.

### Vocabulary Alignment

A critical subtlety: during inference, the test features **must** be extracted using the **same vocabulary** as training. If a test sequence contains a k-mer not seen in training, it must be mapped to the correct column index (or ignored). The code handles this by passing the training vocabulary to `extract_features_weighted` during inference.

When the training vocabulary is the full alphabet permutation (`6^k` entries), the test features are guaranteed to be aligned because every possible k-mer has a column. The code detects this case and avoids redundant computation of the test self-norm.

---

## Key Data Structures

### `train_states` (dict)

Saved during training and persisted in the model artifact. Contains per-k state needed for inference:

```python
{
    3: {
        'k': 3,
        'vocabulary': {'AAA': 0, 'AAC': 1, ...},   # k-mer → column index
        'X_train': <sparse matrix>,                   # Training feature matrix
        'diag_train': array([...])                     # Training self-similarities
    },
    4: { ... },
    5: { ... },
    6: { ... }
}
```
