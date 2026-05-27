# Tumor DNA Novelty Detection — Project Overview

## What This Project Does

This project implements a **cancer detection system from cell-free DNA (cfDNA)** using novelty detection. Instead of training a classifier that needs labeled tumor samples, we train a model that learns what **healthy** DNA looks like, and then flags anything that deviates from that baseline as anomalous.

The core idea: if a patient has cancer, some fraction of their circulating DNA will carry somatic mutations. These mutated sequences will look "novel" compared to a healthy baseline — and we can detect that signal even when the vast majority of the patient's DNA is perfectly normal.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      DATA LAYER                             │
│  FASTA files → MMapFastaReader → sampled DNA sequences      │
│  (src/data_utils.py)                                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    KERNEL LAYER                              │
│  Sequences → k-mer features → Mismatch Kernel → Gram Matrix │
│  Multiple Kernel Learning (MKL) fusion across k values       │
│  (src/kernels.py)                                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   DETECTION LAYER                            │
│  One-Class SVM on precomputed kernel                         │
│  Sequence-level anomaly scores → τ_seq thresholding          │
│  Patient-level aggregation (Outlier Proportion)              │
│  (src/evaluation.py)                                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  CLINICAL DECISION                           │
│  Bootstrap-calibrated patient threshold (τ_pat)              │
│  Binary diagnosis: HEALTHY / TUMOR DETECTED                  │
│  (scripts/calibrate_threshold.py)                            │
└─────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
tumor-dna-novelty-detection/
├── src/                          # Core library
│   ├── data_utils.py             # FASTA I/O, cohort loading
│   ├── kernels.py                # String kernels, MKL, Gram matrices
│   ├── evaluation.py             # OC-SVM fitting, scoring, patient aggregation
│   └── model_io.py               # Model serialization (save/load .pkl)
│
├── experiments/                  # All-in-one experiment runners
│   ├── experiments_utils.py      # Shared CLI args, logging, cohort builders
│   └── run_mil_experiment.py     # End-to-end MIL pipeline (train + eval)
│
├── scripts/                      # Production pipeline (modular steps)
│   ├── train_and_save_svm.py     # Step 1: Train model on healthy baseline
│   ├── calibrate_threshold.py    # Step 2: Calibrate decision threshold
│   ├── run_inference.py          # Step 3a: Score a single patient
│   └── run_full_inference.py     # Step 3b: Score the full test cohort
│
├── models/                       # Saved model artifacts (.pkl)
├── data/                         # FASTA index cache (.fai)
└── tests/                        # Unit tests
```

## The Two Ways to Run the Pipeline

### 1. All-in-One (Research / Experimentation)

```bash
python3 experiments/run_mil_experiment.py \
    --data-dir /path/to/fasta/files \
    --max-train 25000 \
    --max-test-normal 2000 \
    --max-test-tumor 9000
```

This trains, evaluates, and reports Patient-Level ROC-AUC in a single run. Good for rapid iteration but does not save the model.

### 2. Modular Pipeline (Production)

```bash
# Step 1: Train and save model
python3 scripts/train_and_save_svm.py \
    --data-dir /path/to/fasta/files \
    --max-train 25000 \
    --model-name my_model.pkl

# Step 2: Calibrate clinical threshold
python3 scripts/calibrate_threshold.py \
    --data-dir /path/to/fasta/files \
    --model-path models/my_model.pkl

# Step 3: Run inference on a new patient
python3 scripts/run_inference.py \
    --patient-file /path/to/patient.fa \
    --model-path models/my_model.pkl
```

## Key Concepts

| Concept | What It Means | Where It Lives |
|---|---|---|
| **Mismatch String Kernel** | Measures sequence similarity by counting shared k-mers with tolerance for point mutations | `src/kernels.py` |
| **Multiple Kernel Learning (MKL)** | Combines kernels from different k-mer lengths into one weighted kernel | `src/kernels.py` |
| **One-Class SVM** | Learns a boundary around healthy DNA; anything outside is anomalous | `src/evaluation.py` |
| **τ_seq (tau_seq)** | Sequence-level anomaly threshold calibrated at a fixed False Positive Rate | `src/evaluation.py` |
| **Outlier Proportion** | Patient score = fraction of their sequences exceeding τ_seq | `src/evaluation.py` |
| **τ_pat (optimal_threshold)** | Patient-level decision boundary found via bootstrapped Youden's J | `scripts/calibrate_threshold.py` |
| **MIL (Multiple Instance Learning)** | Framework where patients are "bags" of sequence "instances" | Entire pipeline |
| **OA Pipeline (Overlapping Area)** | Distribution-based scoring comparing patient KDEs to healthy KDEs | `src/evaluation_oa.py`, `docs/07-oa-pipeline.md` |

For detailed explanations of each concept, see the individual documentation files in this folder.
