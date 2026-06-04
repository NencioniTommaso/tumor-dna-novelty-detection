# The Overlapping Area (OA) Pipeline

This chapter provides a comprehensive walkthrough of the **Overlapping Area (OA)** novelty detection pipeline. Unlike the older threshold-based method that relied on an explicit One-Class SVM decision boundary, the OA pipeline directly compares distance distributions in kernel space using Kernel Density Estimation (KDE). 

The full process flows from data loading to kernel calculations, KDE fitting, and finally computing the OA score.

---

## 1. Data Loading and Patient Tracking

The OA pipeline begins by loading DNA sequence data while strictly tracking which sequences belong to which patient. This is crucial because novelty detection happens at the sequence level, but the final diagnosis must be aggregated at the patient level.

Using `load_tracked_patient_cohort` (in `src/data_utils.py`), the pipeline:
1. Loads healthy training data as a single baseline "bag" of sequences.
2. Loads test data (healthy and tumor patients), returning `test_files_info` which logs each patient's filename, true label, and exact sequence count.

This structured loading guarantees that later on, after the large kernel matrices are computed, we can slice the matrix precisely to extract one patient's sequences at a time.

---

## 2. Kernel Calculation and Distance Mapping

Like the standard pipeline, we measure sequence similarity using string kernels (e.g., Mismatch Kernel). Because DNA strings are not numerical vectors, we use the kernel trick to implicitly map them into a high-dimensional feature space.

1. **Multiple Kernel Learning (MKL)**: The `compute_mkl_train_test` function computes kernels for varying $k$-mer lengths and combines them into unified training ($K_{train}$) and cross-testing ($K_{test}$) Gram matrices.
2. **Normalization**: The kernel matrices are normalized so that $K(x,x) = 1$ for all sequences. 
3. **From Similarity to Distance**: The OA approach requires *distances*, not just similarities. Given a normalized kernel, the Euclidean distance in the feature space between sequence $x$ and sequence $y$ is computed as:
   
   $$ D^2(x, y) = 2 - 2K(x, y) $$

   This formula provides the foundation for our distance distributions.

---

## 3. Kernel Density Estimation (KDE)

Instead of setting a hard anomaly threshold, the OA method compares the "shape" of two distributions: the baseline healthy distribution and the specific patient's distribution. We approximate these continuous distributions using **Gaussian Kernel Density Estimation (KDE)**.

### The Reference Intra-Distribution (Healthy Baseline)
During the training phase, we compute the **Intra-Distances**: the pairwise distances between all sequences within the healthy training set. We then fit a reference Gaussian KDE to these distances. This single curve represents what a perfectly normal DNA distance profile looks like.

### The Patient Inter-Distribution
During testing, for each patient, we compute the **Inter-Distances**: the distances between that patient's sequences and the healthy training sequences. We fit a new Gaussian KDE specifically for this patient.

> **Note on Downsampling**: A dataset with 20,000 training sequences yields nearly 200 million pairwise intra-distances. To avoid memory exhaustion and excessive computation time, `compute_kde` randomly downsamples the distances (e.g., `max_samples=2,000,000`) before fitting the KDE. This accurately approximates the true distribution while keeping the pipeline fast.

---

## 4. Overlapping Area (OA) and Anomaly Scoring

With the two KDE curves generated over a shared domain (the $x$-axis of possible distances), we measure how much they overlap. 

1. **Compute Area**: We integrate the absolute difference between the reference intra-KDE ($y_{intra}$) and the patient's inter-KDE ($y_{inter}$).
2. **Overlapping Area (OA)**: The raw overlap is computed as:
   
   $$ OA = 1.0 - \frac{\text{Area between curves}}{2.0} $$

   An OA of $1.0$ means the patient's distance distribution is identical to the healthy baseline. A lower OA means the patient's sequences are structurally shifted away from the healthy baseline, indicating the presence of novel, mutated tumor DNA.

3. **Final Anomaly Score**: For intuitive evaluation, the pipeline outputs the anomaly score as:
   
   $$ \text{Score} = 1.0 - OA $$

   Higher scores indicate higher suspicion of cancer.

---

## 5. Evaluation, ROC-AUC, and Plotting (etc.)

Once an OA anomaly score is computed for every patient in the test set, the pipeline performs clinical evaluation:

- **Patient-Level ROC-AUC**: We compute the Area Under the Receiver Operating Characteristic Curve based on the binary patient labels and their continuous $1.0 - OA$ scores. Because this metric operates directly on the true clinical goal (patient diagnosis) rather than sequence-level noise, it provides the true measure of model performance.
- **Visualizations**: The `plotter_oa.py` module generates insightful diagnostic charts:
  - **Single Patient KDEs**: Shows the healthy reference KDE overlaid with the patient's KDE, visually highlighting the shaded overlapping area.
  - **All-Patients Overlay**: A density plot showing all healthy patients bunching closely to the reference curve, while tumor patients diverge.
  - **Score Distributions**: Bar charts separating healthy from tumor patients based on their final OA score.

By replacing hard thresholds with full distributional comparisons, the OA pipeline provides a more robust, parameter-free, and interpretable method for clinical novelty detection.
