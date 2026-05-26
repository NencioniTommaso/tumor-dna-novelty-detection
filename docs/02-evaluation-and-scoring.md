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

## Patient-Level Scoring: Overlapping Area (OA)

### From Sequences to Patients

Each patient contributes hundreds or thousands of DNA sequences. While we can score sequences individually (e.g. via OC-SVM), the core clinical question is about the **patient**: do they have cancer?

Instead of thresholding individual sequence scores, the new pipeline uses a distribution-based approach by computing the **Overlapping Area (OA)** between sequence distance distributions.

### The Aggregation Method

The patient score is calculated by comparing two distributions of sequence distances:
1. **Reference Distribution (Healthy Intra-Distances)**: Pairwise distances between all sequences in the healthy training set.
2. **Patient Distribution (Inter-Distances)**: Distances between the patient's sequences and the healthy training sequences.

Distances are derived directly from the precomputed kernel matrix (assuming a normalized kernel where $K(x,x)=1$):
`D^2(x,y) = K(x,x) + K(y,y) - 2K(x,y) = 2 - 2K(x,y)`

We fit a Gaussian Kernel Density Estimate (KDE) to both sets of distances. The Overlapping Area (OA) is then computed as the area intersected by the two KDE curves. 

```python
def compute_patient_score(patient_K, y_intra, xs, downsample_kde=True):
    # Compute patient vs healthy distances
    patient_inter_distances = compute_distances(patient_K.flatten())
    
    # Fit KDE for patient distances
    _, y_inter = compute_kde(
        patient_inter_distances, xmax=xs[-1], num_points=len(xs), downsample=downsample_kde
    )
    
    # Compute Overlapping Area
    area = np.trapezoid(np.abs(y_intra - y_inter), xs)
    oa = 1.0 - area / 2.0
    
    # Final anomaly score (Higher score = more anomalous)
    return 1.0 - oa
```

This returns an anomaly score between 0.0 and 1.0.

### The Downsample Parameter

Because the number of pairwise distances grows quadratically (e.g., thousands of sequences result in millions of distances), fitting a KDE on the full set is computationally prohibitive. 

To address this, we use a `downsample` parameter in the KDE computation. When `downsample=True`, we randomly sample up to a maximum number of distances (defaulting to 2,000,000). We explicitly tried to change the downsample parameter in order to have a good approximation of the real value without slowing down the pipeline unnecessarily. This specific value guarantees that the KDE accurately reflects the true distance distribution while avoiding out-of-memory errors or extreme slowdowns.

### Why Overlapping Area > Outlier Proportion

The previous method ("Outlier Proportion") relied on calculating the proportion of sequences exceeding a strict anomaly threshold (`τ_seq`). The OA method improves upon this because:
- **Distributional Robustness**: By comparing entire distributions rather than relying on a hard cutoff point, it is highly robust to noise and varying tumor fractions.
- **Threshold-free**: We no longer need to calibrate a sequence-level threshold (`τ_seq`) on the training set.
- **Unit-free Measure**: The OA provides a robust metric that is strictly bounded between 0.0 and 1.0.

---

## Sequence-Level vs. Patient-Level ROC-AUC

The pipeline reports two AUC values:

### Sequence-Level AUC (noisy, expected to be low)

Every sequence from a tumor patient is labeled as `-1` (cancer), even though most of them are actually normal germline DNA. This means the sequence-level labels are **systematically wrong** — a tumor patient with 5% ctDNA has 95% healthy sequences mislabeled as cancer. The sequence-level AUC reflects this noise and is not clinically meaningful.

### Patient-Level AUC (the real metric)

After aggregation, each patient gets a single score. The labels are now correct: the patient either has cancer or doesn't. This AUC is the true measure of diagnostic performance.
