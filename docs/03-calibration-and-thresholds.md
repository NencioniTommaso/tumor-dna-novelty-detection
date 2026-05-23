# Threshold Calibration and Clinical Decision

> **File**: `scripts/calibrate_threshold.py`
> This script bridges the gap between raw anomaly scores and clinical diagnoses by finding an optimal patient-level decision threshold using statistical bootstrapping.

## The Two-Threshold System

The pipeline uses two thresholds in sequence:

```
DNA Sequence ──→ Anomaly Score ──→ τ_seq ──→ Outlier? (yes/no)
                                      │
    ┌─────────── per patient ──────────┘
    │
    ▼
Outlier Proportion ──→ τ_pat ──→ Diagnosis (HEALTHY / TUMOR)
```

| Threshold | Name in Code | Set During | Purpose |
|---|---|---|---|
| **τ_seq** | `tau_seq` | Training (`train_and_save_svm.py`) | Declares individual sequences as outliers |
| **τ_pat** | `optimal_threshold` | Calibration (`calibrate_threshold.py`) | Declares patients as tumor-positive |

τ_seq is a statistical constant (99th percentile of healthy training scores). τ_pat is a clinical decision boundary that must be tuned on labeled validation data.

---

## Why Calibration Requires a Separate Step

The training phase uses **only** healthy data — we intentionally withhold tumor samples to avoid contaminating the novelty detector's concept of "normal." But to set a diagnostic threshold, we need to know where healthy patients end and tumor patients begin. This requires a **validation cohort with both classes**.

The calibration script:
1. Loads the pretrained model (no retraining).
2. Scores a validation cohort of known healthy and tumor patients.
3. Finds the optimal decision boundary between them.

---

## Youden's J Statistic

The optimal threshold is chosen to maximize **Youden's J statistic**:

```
J = Sensitivity + Specificity - 1
  = TPR - FPR
```

At every possible threshold along the ROC curve, J measures the combined ability to correctly identify both tumor patients (sensitivity) and healthy patients (specificity). The threshold that maximizes J is the point on the ROC curve farthest from the diagonal (random chance).

```
    1.0 ┤          ╭──────────
        │        ╭─╯
  TPR   │      ╭─╯           ← Youden's J = max vertical
        │    ╭─╯                distance from diagonal
        │  ╭─╯
        │╭─╯
    0.0 ┼──────────────────
       0.0      FPR       1.0
```

---

## Stratified Bootstrap

With only ~10 patients per class (typical in this dataset), a single ROC curve is unstable. The calibration script uses **stratified bootstrapping** to get a robust estimate:

```python
for _ in range(1000):
    # Resample healthy patients WITH replacement
    boot_healthy = rng.choice(idx_healthy, size=len(idx_healthy), replace=True)
    # Resample tumor patients WITH replacement
    boot_tumor = rng.choice(idx_tumor, size=len(idx_tumor), replace=True)
    
    # Compute Youden's J on this resampled cohort
    fpr, tpr, thresholds = roc_curve(resampled_binary, resampled_scores)
    best_threshold = thresholds[argmax(tpr - fpr)]
    collected_thresholds.append(best_threshold)
```

### Why Stratified?

Standard bootstrap randomly resamples the full dataset. With 2 healthy and 9 tumor patients, you could get a bootstrap sample with 0 healthy patients — making the ROC curve undefined. **Stratified** bootstrap resamples within each class independently, guaranteeing both classes are always represented.

### The Final Threshold

```python
optimal_threshold = np.median(collected_thresholds)
ci_lower = np.percentile(collected_thresholds, 2.5)
ci_upper = np.percentile(collected_thresholds, 97.5)
```

The **median** of N bootstrap thresholds is taken as the robust estimate. The 2.5th and 97.5th percentiles give a 95% confidence interval, which quantifies uncertainty from the small sample size.

---

## Model Artifact Update

After calibration, the optimal threshold is written **back into the model artifact**:

```python
saved_state = joblib.load(model_path)
saved_state['optimal_threshold'] = optimal_threshold
joblib.dump(saved_state, model_path)
```

This means the `.pkl` file is self-contained — the inference script can load it and immediately make binary diagnoses without needing to know the calibration history.

---

## The Inference Decision

In `run_inference.py`, the calibrated threshold is used for a binary decision:

```python
if patient_score >= optimal_threshold:
    diagnosis = "🚨 TUMOR DETECTED"
else:
    diagnosis = "✅ HEALTHY"
```

If the model has not been calibrated yet (`optimal_threshold is None`), the script warns the user and outputs only the raw score without a diagnosis.
