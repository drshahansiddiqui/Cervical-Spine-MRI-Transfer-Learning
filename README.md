# Transfer Learning for Multi-Label Multi-Class Classification of Cervical Spine Disorders on MRI

---

## 1. Description

This project classifies cervical spine disorders on magnetic resonance imaging (MRI) with a transfer-learning feature-fusion framework:

1. Each MRI slice is preprocessed: intensity windowing, background crop, denoising, square padding, 224 × 224 resize, and grayscale → RGB.
2. Frozen ImageNet-pretrained **ResNet50** (2,048-d) and **VGG16** (512-d) extract global-average-pooled features. These are concatenated into a **2,560-d fused vector**.
3. A panel of **class-balanced classifiers** is screened for every task: logistic regression, extra trees, random forest, and an RBF-SVM on PCA-reduced features.
4. For binary tasks, the **decision threshold is tuned on a validation subset** instead of using a fixed 0.50 cut-off.
5. The selected model and threshold are evaluated **once** on a held-out test subset.

The framework is evaluated in four diagnostic settings covering nine tasks:

| Part | Setting | Tasks |
|------|---------|-------|
| A | Binary (one disease vs. healthy) | Healthy vs. myelopathy, Healthy vs. cord edema, Healthy vs. disc herniation, Healthy vs. unhealthy |
| B | Multi-label binary (disease combination vs. healthy) | Healthy vs. disc bulging + disc herniation, Healthy vs. myelopathy + disc herniation, Healthy vs. cord edema + disc herniation |
| C | Multi-class (4 classes) | Healthy, myelopathy, cord edema, disc herniation |
| D | Multi-label multi-class (6 classes) | Healthy plus supported disease-combination classes |

---

## 2. Dataset Information

| Item | Details |
|------|---------|
| Name | Cervical Spine MRI Dataset (multi-label) |
| Repository | Zenodo |
| DOI | https://doi.org/10.5281/zenodo.23297455 |
| Origin | Locally curated, retrospectively collected clinical MRI studies, de-identified before release (patients are identified only by numeric IDs between 1 and 63; IDs 5 and 55 are not used) |
| Ethics approval | Research Ethics Institutional Review Board, Lahore College for Women University (Certificate No. ORIC/LCWU/238). The requirement for patient consent was waived because the data are retrospective and de-identified. |
| Modality | Cervical spine MRI (sagittal and axial slices), provided as 8-bit 192 × 192 JPEG images with no embedded metadata |
| Size | 1,464 slices from 61 patients (24 slices per patient) |
| Annotation | Patient-level multi-label diagnosis assigned to every slice of that patient; healthy controls carry no disease label |
| License | Creative Commons Attribution 4.0 International (CC BY 4.0) |

**Patients and label instances per category:**

| Category | Patients | Label instances (slices) | Used in reported results |
|----------|----------|--------------------------|--------------------------|
| Healthy | 18 | 432 | Yes |
| Disc bulging | 41 | 984 | Part B combinations only |
| Myelopathy | 4 | 96 | Yes |
| Cord edema | 4 | 96 | Yes |
| Disc herniation | 4 | 96 | Yes |
| Stenosis | 1 | 24 | No |
| Scoliotic deformity | 1 | 24 | No |
| Osteophytosis | 1 | 24 | No |
| Spondylodiscitis | 1 | 24 | No |

A patient can have more than one diagnosis, so the disease rows add up to more than 61 patients.

### Expected data layout
```
MyDrive/CervicalSpine/processed/
├── manifest.csv
├── 1/
│   ├── 1_sl000.jpg
│   ├── ...
│   └── 1_sl023.jpg
├── 2/
└── ... 63/        (61 patient folders)
```

`manifest.csv` has one row per slice. A copy is also in `data/manifest.csv` in this repository.

| Column | Meaning |
|--------|---------|
| `path` | Image path in Google Colab (`/content/drive/MyDrive/CervicalSpine/processed/<id>/<id>_slNNN.jpg`) |
| `patient_id` | De-identified patient number |
| `is_normal` | 1 = healthy control |
| `Disc Buldging`, `Myleopathy`, `Stenosis`, `Cord Edma`, `Scoliotic Deformity`, `Disc Herniation`, `Osteophytosis`, `Spondylodiscitis` | 0/1 disease labels. The column names keep the original spellings used in the code. |

---

## 3. Code Information

| File | Purpose | Main outputs (under `MyDrive/CervicalSpine/`) |
|------|---------|-----------------------------------------------|
| `01_preprocessing_and_feature_extraction.ipynb` | **Step 1.** Records the computing environment, loads the manifest, preprocesses and caches all 1,464 slices, and extracts the fused ResNet50 + VGG16 features | `dual_eval_output/prep_cache/cached_manifest.csv`, `dual_eval_output/prep_cache/images/`, `dual_eval_output/feature_cache/concat_<hash>.npz` (1,464 × 2,560) |
| `02_classification_and_results.ipynb` | **Step 2.** 2a screens all eight categories; 2b runs the nine reported tasks (Table 3); 2c generates the article figures and tables; 2d (optional) produces full-dataset confusion matrices for visualization only | `dual_eval_output_v50_improved_existing_cache/`, `dual_eval_output_v51_selected_reportable_classes/`, `FINAL_CODE_SIDE_ARTIFACTS_V52/`, `FULL_DATASET_CONFUSION_MATRICES_V51/` |
| `data/manifest.csv` | Labels and patient IDs for all slices (same file as in the Zenodo dataset) | |
| `requirements.txt` | Python dependencies | |
| `LICENSE` | MIT license | |

The notebooks keep the outputs of the original runs, so the printed results can be checked without re-running them.

**Note on cell 2d:** it refits the selected model on train + validation and predicts **every** image, including training images. Its confusion matrices are for visualization only and are **not** held-out results.

---

## 4. Requirements

### Python libraries
```
tensorflow==2.20.0
numpy
pandas
scikit-learn
matplotlib
opencv-python-headless
Pillow
openpyxl
```
Google Colab already includes all of these. To install them elsewhere:
```bash
pip install -r requirements.txt
```

### Computing infrastructure
| Component | Specification |
|-----------|---------------|
| Platform | Google Colab Pro, T4 GPU runtime with standard RAM |
| Operating system | Ubuntu Linux, kernel 6.6 (Colab runtime, x86_64) |
| CPU | Intel Xeon @ 2.00 GHz, 2 virtual cores |
| RAM | 12.7 GB system memory |
| GPU | NVIDIA Tesla T4, 15 GB GPU memory |
| Language / framework | Python 3, TensorFlow 2.20.0 (Keras API), scikit-learn |
| Storage | Google Drive |

The article's experiments ran with TensorFlow 2.20.0, as printed in the saved notebook outputs. The same Colab T4 machine type was later checked with numpy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1, OpenCV 4.14.0, matplotlib 3.10.0, Pillow 11.3.0 and openpyxl 3.1.5.

The GPU is used only for the one-time ResNet50/VGG16 feature extraction in Step 1. Step 2 runs on the CPU.

---

## 5. Usage Instructions

1. **Download the dataset** from Zenodo (`cervical_spine_mri_dataset.zip`) and unzip it: https://doi.org/10.5281/zenodo.23297455
2. **Copy it to Google Drive** so the images and manifest are at `MyDrive/CervicalSpine/processed/` (see the layout in Section 2).
3. **Run Step 1.** Open `01_preprocessing_and_feature_extraction.ipynb` in Google Colab, select *Runtime → Change runtime type → T4 GPU*, then click *Runtime → Run all*. When Colab asks, allow access to Google Drive.
4. **Run Step 2.** Open `02_classification_and_results.ipynb` and click *Runtime → Run all*. No GPU is needed.
5. **Read the results:**
   - `MyDrive/CervicalSpine/dual_eval_output_v51_selected_reportable_classes/V51_SELECTED_BEST_PER_TASK.csv` holds the final metrics for the nine tasks (Table 3 of the article).
   - `MyDrive/CervicalSpine/FINAL_CODE_SIDE_ARTIFACTS_V52/` holds the figures (confusion matrices, ROC curves, bar charts), tables and sample images.

**Running outside Colab:** remove the `from google.colab import drive` and `drive.mount(...)` lines. Then change the `/content/drive/MyDrive/CervicalSpine` paths, both in the notebooks' configuration and in the `path` column of `manifest.csv`, to your local folder.

**Reproducibility:** all random operations use seed 42. Feature extraction on a different GPU or library version can change floating-point values very slightly, so re-run metrics may differ from the article in the last decimal places.

### Fixed settings
| Setting | Value |
|---------|-------|
| Random seed | 42 |
| Split | Image-level, stratified by class: 64% train / 16% validation / 20% test |
| Input size | 224 × 224 × 3 |
| Fused feature size | 2,560 (2,048 ResNet50 + 512 VGG16) |
| PCA components (SVM only) | 128 |
| Logistic regression | C ∈ {0.03, 0.1, 0.3, 1, 3}, class_weight = balanced |
| RBF-SVM | C ∈ {1, 3, 10}, gamma = scale, class_weight = balanced |
| Random forest / extra trees | 700 trees, max_features = sqrt, class_weight = balanced |
| Threshold sweep | 91 values from 0.05 to 0.95 (step 0.01); selection by validation F1 or balanced accuracy |
| Minimum class support | 20 images per class (multi-class tasks) |

---

## 6. Methodology

1. **Preprocessing (image domain, run once and cached as JPEG at quality 95):**
   1. Intensity windowing: values are clipped to the slice's 1st–99th percentile, rescaled to [0, 1], and stored as 8-bit.
   2. Background removal: the slice is cropped to its foreground bounding box (pixels above the 12th-percentile intensity) with an 8% margin.
   3. Denoising: a 3 × 3 median filter, then a bilateral filter (d = 5, σ_color = 35, σ_space = 35).
   4. The slice is zero-padded to a square to keep the aspect ratio.
   5. The slice is resized to 224 × 224 with area interpolation.
   6. The grayscale channel is replicated to three RGB channels.
2. **Backbone input normalization (at load time):** each image goes through the Keras `preprocess_input` of its backbone (ImageNet mean subtraction).
3. **Feature extraction:** frozen ResNet50 and VGG16 with global average pooling produce 2,048-d and 512-d features, which are concatenated into a 2,560-d vector. No data augmentation is used.
4. **Class selection:** all eight categories are first screened as one-vs-healthy tasks (Step 2a). Myelopathy, cord edema and disc herniation are retained. Disc bulging is kept only in the combination tasks, because as the dominant class it would otherwise swamp the multi-class settings. The four categories with 24 slices each are excluded.
5. **Data splitting:** stratified image-level split, 64/16/20, seed 42.
6. **Classifier screening:** class-weighted classifiers with w_c = N / (K · n_c). The best model is chosen on the validation subset.
7. **Threshold tuning (binary tasks only):** τ* is swept over [0.05, 0.95] on validation data. Multi-class tasks use the argmax.
8. **Testing:** the selected model and τ* are applied once to the held-out test subset.

### Evaluation method
- **Held-out testing:** every reported figure comes from the 20% test subset, which was used neither for training nor for model or threshold selection.
- **Baseline check:** each test accuracy is compared with the majority-class baseline of its task.
- **Fixed vs. tuned threshold:** results at the default 0.50 cut-off are compared with results at the validation-tuned threshold.
- **Comparison with state-of-the-art published methods:** results are compared with Zhang et al. (ROI + cascade CNN), Rhee et al. (four-backbone ensemble), Song et al. (cascade-ensemble multi-task network), Wang et al. (hybrid feature-extraction network), Abuhayi et al. (Inv-AlxVGGNets), Xie et al. (radiomics + MedSAM) and Deepa & Paul (feature-level fusion). The comparison covers accuracy, AUC, multi-class/multi-label capability, and the use of class balancing and threshold tuning (Table 6 of the article).

### Assessment metrics
| Metric | Why it is used |
|--------|----------------|
| Accuracy | Overall proportion of correct predictions. Easy to interpret, but it can be inflated by the majority class, so it is always reported with the metrics below. |
| Balanced accuracy | Mean recall across classes. Used for model and threshold selection because the classes are strongly imbalanced. |
| Precision | Of the images flagged as diseased, the fraction that are truly diseased. Measures false alarms. |
| Recall (sensitivity) | Of the truly diseased images, the fraction detected. Clinically critical, because a missed disorder delays treatment. |
| F1-score | Harmonic mean of precision and recall. Robust under class imbalance. |
| AUC-ROC | Threshold-independent discrimination ability, computed with the trapezoidal rule. Uses one-vs-rest macro averaging for multi-class tasks. |
| Confusion matrix | Shows class-wise error patterns that aggregate metrics hide. |

For multi-class tasks, precision, recall, F1 and AUC are **macro-averaged**, so small classes count equally with large ones.

### Main results (held-out test subset)
| Part | Task | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) | AUC |
|------|------|------|------|------|------|------|
| A | Healthy vs. myelopathy | 93.40 | 80.00 | 84.21 | 82.05 | 0.964 |
| A | Healthy vs. cord edema | 91.51 | 75.00 | 78.95 | 76.92 | 0.901 |
| A | Healthy vs. disc herniation | 90.57 | 71.43 | 78.95 | 75.00 | 0.920 |
| A | Healthy vs. unhealthy | 85.67 | 86.67 | 94.20 | 90.28 | 0.851 |
| B | Healthy vs. disc bulging + disc herniation | 94.06 | 70.00 | 100.00 | 82.35 | 0.979 |
| B | Healthy vs. myelopathy + disc herniation | 96.88 | 88.89 | 80.00 | 84.21 | 0.965 |
| B | Healthy vs. cord edema + disc herniation | 96.88 | 88.89 | 80.00 | 84.21 | 0.965 |
| C | Four-class | 92.45 | 97.89 | 73.33 | 80.15 | 0.921 |
| D | Six-class | 86.67 | 85.97 | 73.30 | 78.23 | 0.956 |

---

## 7. Conclusions

- A frozen ResNet50 + VGG16 feature-fusion framework with class-balanced classifier screening and validation-tuned thresholds discriminates cervical spine disorders reliably across four increasingly demanding diagnostic settings.
- Held-out accuracy reaches 96.88% on the multi-label binary tasks and 93.40% on single-disease binary screening. It holds at 92.45% and 86.67% on the four-class and six-class settings, with AUC between 0.851 and 0.979.
- The validation-tuned thresholds (0.11–0.53) depart substantially from the default 0.50 in several tasks. Reporting the operating point together with class-wise metrics therefore matters as much as the architecture itself.
- Broad diagnostic scope (nine tasks, multi-label and multi-class) and class-sensitive, threshold-aware evaluation can be achieved together within a lightweight transfer-learning pipeline that needs no fine-tuning.

---

## 8. Limitations

- **Image-level split:** slices from the same patient can appear in both the training and test subsets, so the reported metrics are optimistic compared with a patient-grouped evaluation. Patient IDs are tracked, and the number of distinct test patients is reported for every task.
- **Few patients per disease:** each retained disease category (myelopathy, cord edema, disc herniation) comes from only four patients, which limits how well the results generalize.
- **Excluded categories:** stenosis, scoliotic deformity, osteophytosis and spondylodiscitis have only one patient (24 slices) each, too few for reliable per-class estimates, so they are not part of the reported results.
- **Patient-level labels:** each patient's diagnosis is assigned to all of that patient's slices, including slices where the finding may not be visible, which introduces label noise.
- **Single-center, retrospective data:** all images come from one local cohort. Generalization to other scanners, protocols and institutions has not been tested.
- **No external or prospective validation:** the framework must be validated on independent cohorts, and prospectively, before any clinical use.

---

## 9. Citations

**Code:**
```
Rehman, U., & Siddiqui, S. Y. (2026). drshahansiddiqui/Cervical-Spine-MRI-Transfer-Learning:
First Release (v1.1) (Version v1.1) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.23297698
GitHub: https://github.com/drshahansiddiqui/Cervical-Spine-MRI-Transfer-Learning
```

**Dataset:**
```
Siddiqui, S. Y., & Rehman, U. (2026). A Multi-Label Annotated Cervical Spine MRI Dataset
for Multi Label Multi Class Classification of Degenerative and Structural Disorders [Dataset].
Zenodo. https://doi.org/10.5281/zenodo.23297455
```

---

## 10. License

- **Code:** MIT License; see `LICENSE`.
- **Dataset:** distributed separately on Zenodo under the CC BY 4.0 license.
