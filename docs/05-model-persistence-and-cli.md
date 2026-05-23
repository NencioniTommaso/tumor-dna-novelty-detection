# Model Persistence and the `.pkl` Artifact

> **File**: `src/model_io.py`
> This module handles saving and loading trained model artifacts using `joblib`.

## What Gets Saved

The model artifact is a single `.pkl` file (e.g., `models/ocsvm_pretrained.pkl`) containing everything needed for inference:

```python
state = {
    'model':            svm,              # Fitted OneClassSVM object
    'train_sequences':  train_data,       # Raw training sequences (for reference)
    'max_k':            6,                # Maximum k-mer length
    'mismatches':       1,                # Mismatch tolerance
    'nu_param':         0.2,              # SVM nu parameter used during training
    'mkl_weights':      [0, 0, 0.1, ...], # MKL weight vector
    'train_states':     { ... },          # Per-k feature matrices and vocabularies
    'tau_seq':          0.0342,           # Sequence-level anomaly threshold
    'optimal_threshold': None,            # Patient-level threshold (set by calibration)
}
```

### Why Save `train_states`?

The precomputed kernel SVM needs to compute `K_test = X_test @ X_train.T` at inference time. This requires `X_train` — the sparse feature matrix of the training sequences. Rather than re-extracting features from raw sequences every time, the training feature matrices are saved directly.

### Why Save `train_sequences`?

Primarily for debugging and audit trails. They are not used during inference.

### The `optimal_threshold` Field

This field starts as `None` when the model is first trained. It is populated later by `calibrate_threshold.py`, which reads the artifact, adds the threshold, and writes it back. This two-phase approach separates training (unsupervised, healthy-only) from calibration (supervised, both classes).

---

## CLI Arguments Reference

All scripts use a shared argument system defined in `experiments/experiments_utils.py`.

### Data Arguments

| Argument | Default | Description |
|---|---|---|
| `--data-dir` | *(required)* | Path to directory containing FASTA files |
| `--cache-dir` | `data/.fai_cache` | Path for FASTA index cache files |

### Sampling Arguments

| Argument | Default | Description |
|---|---|---|
| `--max-train` | 18000 | Total sequences to sample for training (split evenly across files) |
| `--max-test-normal` | 1500 | Total healthy sequences for testing |
| `--max-test-tumor` | 1500 | Total tumor sequences for testing |
| `--sample-size` | 1500 | Sequences to sample for single-patient inference |
| `--seed` | 42 | Random seed for reproducible sampling |

### Kernel Arguments

| Argument | Default | Description |
|---|---|---|
| `--max-k` | 6 | Maximum k-mer length for the Mixed String Kernel |
| `--mismatches` | 1 | Allowed Hamming distance for mismatch neighborhoods |

### Model Arguments

| Argument | Default | Description |
|---|---|---|
| `--nu-param` | 0.2 | OC-SVM nu parameter (upper bound on outlier fraction) |
| `--seq-fpr` | 0.01 | Sequence-level False Positive Rate for τ_seq calibration |
| `--model-name` | `ocsvm_pretrained.pkl` | Filename for saving/loading model artifacts |
| `--model-path` | `models/ocsvm_pretrained.pkl` | Full path to model artifact |

### Execution Arguments

| Argument | Default | Description |
|---|---|---|
| `--n-jobs` | -1 | Number of CPU cores (-1 = all available) |
