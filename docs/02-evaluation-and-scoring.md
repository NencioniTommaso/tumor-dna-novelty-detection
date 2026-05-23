# Novelty Detection and Patient Scoring

> **File**: `src/evaluation.py`
> This module handles fitting the One-Class SVM, computing sequence-level anomaly scores, and aggregating those scores to patient-level diagnoses using absolute thresholding.

## One-Class SVM: The Anomaly Detector

### What It Is

A **One-Class SVM (OC-SVM)** is trained *only* on healthy data. It learns a decision boundary that encloses the healthy sequences in kernel space. Anything that falls outside this boundary is flagged as anomalous.

```
           Kernel Feature Space
        ┌──────────────────────┐
        │  ·  · ·  ·           │
        │ · · · · · ·          │    · = healthy sequences
        │  · · ·OC-SVM· ·      │    ✕ = tumor sequences
        │ · · boundary· · ·    │
        │  · ·  · · ·  ·       │
        │   ·  · · ·           │
        └──────────────────────┘
                    ✕  ✕
                ✕       ✕
```

### The `nu` Parameter

The `nu` parameter (ν) controls how tight the boundary is:
- **`nu = 0.005`**: Very tight. Only 0.5% of training data is allowed outside the boundary. Conservative — few false positives but may miss subtle tumors.
- **`nu = 0.2`**: Looser. Up to 20% of training data may fall outside. More sensitive to anomalies but higher false positive rate.

`nu` is an upper bound on the fraction of training points that become support vectors (outliers). In practice, it acts as a prior on the expected contamination rate.

### Precomputed Kernel

The SVM operates on a **precomputed kernel matrix**, not raw features. This means:
- `svm.fit(K_train)` receives an N×N matrix of pairwise similarities.
- `svm.predict(K_test)` receives an M×N matrix (test vs. train similarities).
- `svm.decision_function(K_test)` returns a score per test sequence.

This is necessary because we use a custom string kernel that cannot be expressed as a standard kernel function (linear, RBF, etc.).

### Decision Function Scores

The OC-SVM's `decision_function` returns:
- **Positive values** → sequence is inside the healthy boundary (normal).
- **Negative values** → sequence is outside the boundary (anomalous).
- **Magnitude** → how far from the boundary (confidence).

We **invert** these scores (`-score`) so that higher values mean more anomalous. This makes the downstream logic more intuitive: a higher score = more suspicious.

---

## τ_seq: The Sequence-Level Threshold

### The Problem

Not all sequences flagged by the SVM are actually from tumor DNA. Healthy sequences naturally contain noise — sequencing errors, benign polymorphisms, degradation artifacts. The SVM will assign high anomaly scores to some of these purely by chance.

### The Solution: Absolute Thresholding

τ_seq (tau_seq) is a threshold that answers: *"How anomalous does a sequence need to be before we call it a suspected tumor read?"*

It is calibrated on the **healthy training data** at a fixed False Positive Rate (FPR):

```python
train_scores = -svm.decision_function(K_train)   # Invert: higher = more anomalous
tau_seq = np.percentile(train_scores, 100 * (1 - seq_fpr))
```

With the default `seq_fpr = 0.01` (1% FPR), τ_seq is the **99th percentile** of training anomaly scores. This guarantees:

> *"Exactly 1% of sequences from a perfectly healthy person will cross this threshold purely by random noise."*

### The `seq_fpr` Parameter

| `seq_fpr` | Percentile | Meaning |
|---|---|---|
| 0.001 | 99.9th | Very strict. Almost no healthy sequences are flagged. |
| 0.01 | 99th | **Default.** 1% baseline noise expected per healthy patient. |
| 0.05 | 95th | Permissive. 5% of healthy sequences are flagged as noise. |

Lower values make the system more specific (fewer false alarms) but less sensitive (may miss early-stage tumors with very low tumor fractions).

---

## Patient-Level Scoring: Outlier Proportion

### From Sequences to Patients

Each patient contributes hundreds or thousands of DNA sequences. The SVM scores each sequence individually, but the clinical question is about the **patient**: do they have cancer?

This is a **Multiple Instance Learning (MIL)** problem:
- A patient is a **bag** of sequence **instances**.
- The bag is positive (tumor) if *some* of its instances are anomalous.

### The Aggregation Method

The patient score is the **proportion of sequences that exceed τ_seq**:

```python
def compute_patient_score(seq_scores, tau_seq):
    inverted_scores = -np.asarray(seq_scores)
    return float(np.mean(inverted_scores > tau_seq))
```

This returns a value between 0.0 and 1.0:

### Why Outlier Proportion > Top-K Averaging

The previous method averaged the raw anomaly scores of the top 5% most anomalous sequences. This had issues:
- **Sensitive to tumor fraction**: If a patient had only 1% tumor DNA, the top 5% average diluted the signal with 4% healthy noise.
- **Magnitude-dependent**: Raw scores depend on kernel normalization, making cross-experiment comparisons fragile.

Outlier Proportion fixes both:
- It dynamically adapts to any tumor fraction.
- The output is a unit-free proportion, always between 0 and 1.

---

## Sequence-Level vs. Patient-Level ROC-AUC

The pipeline reports two AUC values:

### Sequence-Level AUC (noisy, expected to be low)

Every sequence from a tumor patient is labeled as `-1` (cancer), even though most of them are actually normal germline DNA. This means the sequence-level labels are **systematically wrong** — a tumor patient with 5% ctDNA has 95% healthy sequences mislabeled as cancer. The sequence-level AUC reflects this noise and is not clinically meaningful.

### Patient-Level AUC (the real metric)

After aggregation, each patient gets a single score. The labels are now correct: the patient either has cancer or doesn't. This AUC is the true measure of diagnostic performance.
