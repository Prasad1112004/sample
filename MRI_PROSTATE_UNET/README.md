# AI-Based Prostate MRI Image Segmentation
## Using a Combined U-Net Deep Learning Model with Five Datasets

---

## 📋 Project Overview

This project implements an automated whole-prostate segmentation system using the **U-Net** deep learning architecture. A single combined model is trained on **five publicly available prostate MRI datasets** and evaluated using **eight clinical performance metrics**.

---

## 🗂️ Project Structure

```
MRI_PROSTATE_UNET/
│
├── datasets/                         ← Raw dataset folders (download separately)
│   ├── I2CVB/
│   ├── MSD/
│   ├── NCI_ISBI_3T/
│   ├── NCI_ISBI_Dx/
│   └── PROSTATEx/
│
├── preprocessing/                    ← Dataset-specific preprocessing modules
│   ├── i2cvb_preprocessing.py
│   ├── msd_preprocessing.py
│   ├── nci_3t_preprocessing.py
│   ├── nci_dx_preprocessing.py
│   └── prostatex_preprocessing.py
│
├── models/
│   ├── unet.py                       ← U-Net architecture + loss functions
│   └── combined_prostate_unet.pth   ← Saved model (generated after training)
│
├── training/
│   └── train.py                      ← Combined training pipeline
│
├── evaluation/
│   ├── metrics.py                    ← Eight evaluation metrics
│   └── evaluate.py                   ← Full evaluation + visualisation
│
├── results/
│   ├── graphs/                       ← Generated performance graphs
│   ├── predictions/                  ← Sample segmentation overlays
│   └── metrics.csv                   ← Per-patient evaluation results
│
├── MRI_PROSTATE_UNET_Colab.ipynb    ← Google Colab notebook
├── app.py                            ← Streamlit web application
├── requirements.txt
└── README.md
```

---

## 📦 Datasets

| Dataset | Scanner | Approx. Patients | Format | Annotation |
|---------|---------|-----------------|--------|-----------|
| **I2CVB** | T2w MRI | 19 | NIfTI | Whole gland |
| **MSD Task05** | T2w + ADC | 32 | NIfTI | PZ+TZ → merged |
| **NCI-ISBI 3T** | 3T T2w | 60 | DICOM+NRRD | CG+PZ → merged |
| **NCI-ISBI Dx** | 1.5T T2w | 60 | DICOM+NRRD | CG+PZ → merged |
| **PROSTATEx** | T2w MRI | 204 | DICOM+SEG | Whole gland (Seg-HiRes) |

### Dataset Access Links
- **I2CVB**: https://i2cvb.github.io/
- **MSD**: http://medicaldecathlon.com/
- **NCI-ISBI**: https://wiki.cancerimagingarchive.net/display/Public/NCI-ISBI+2013+Challenge
- **PROSTATEx**: https://wiki.cancerimagingarchive.net/display/Public/PROSTATEx

---

## ⚙️ Installation

### Local Setup
```bash
# 1. Clone / download the project
cd MRI_PROSTATE_UNET

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt
```

### Google Colab
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install extra libraries
!pip install nibabel SimpleITK pynrrd highdicom torchinfo streamlit
```

---

## 🚀 Execution Steps

### Step 1 – Dataset Preprocessing

Run each preprocessing script after placing datasets in the `datasets/` folder:

```bash
# I2CVB
python preprocessing/i2cvb_preprocessing.py \
    --dataset_root datasets/I2CVB \
    --output_root  preprocessed/I2CVB

# MSD Task05
python preprocessing/msd_preprocessing.py \
    --dataset_root datasets/MSD/Task05_Prostate \
    --output_root  preprocessed/MSD

# NCI-ISBI 3T
python preprocessing/nci_3t_preprocessing.py \
    --dataset_root datasets/NCI_ISBI_3T \
    --output_root  preprocessed/NCI_ISBI_3T

# NCI-ISBI Dx
python preprocessing/nci_dx_preprocessing.py \
    --dataset_root datasets/NCI_ISBI_Dx \
    --output_root  preprocessed/NCI_ISBI_Dx

# PROSTATEx
python preprocessing/prostatex_preprocessing.py \
    --dataset_root datasets/PROSTATEx \
    --output_root  preprocessed/PROSTATEx
```

**Expected output**: Each patient folder in `preprocessed/<DATASET>/` will contain:
- `image_slice_XXXX.npy` – 256×256 float32 T2w slice
- `mask_slice_XXXX.npy`  – 256×256 uint8 binary mask
- `meta.json`            – Metadata (spacing, n_slices, etc.)
- `verification.png`     – Visual check (MRI | Mask | Overlay)

### Step 2 – Model Training

```bash
python training/train.py \
    --preprocessed_root preprocessed/ \
    --output_dir        results/ \
    --epochs 30 \
    --batch_size 8 \
    --lr 0.001 \
    --patience 6
```

**GPU acceleration** (recommended):
```bash
python training/train.py --preprocessed_root preprocessed/ --amp
```

**Expected output**:
- `results/models/combined_prostate_unet.pth` – Best checkpoint
- `results/training_history.json`
- `results/split_train/val/test.csv`
- `results/graphs/training_curves.png`

### Step 3 – Model Evaluation

```bash
python evaluation/evaluate.py \
    --checkpoint  results/models/combined_prostate_unet.pth \
    --test_csv    results/split_test.csv \
    --output_dir  results/
```

**Expected output**:
- `results/metrics.csv`             – Per-patient 8 metrics
- `results/graphs/`                 – All performance graphs
- `results/predictions/`            – Segmentation overlays

### Step 4 – Streamlit Web App

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

### Step 5 – Google Colab (All-in-one)

Open `MRI_PROSTATE_UNET_Colab.ipynb` in Google Colab and run all cells sequentially.

---

## 🏗️ U-Net Architecture

```
Input: (B, 1, 256, 256)

Encoder:
  Level 1: DoubleConv(1→32)    → MaxPool   [skip1: 32×256×256]
  Level 2: DoubleConv(32→64)   → MaxPool   [skip2: 64×128×128]
  Level 3: DoubleConv(64→128)  → MaxPool   [skip3: 128×64×64]
  Level 4: DoubleConv(128→256) → MaxPool   [skip4: 256×32×32]

Bottleneck:
  DoubleConv(256→512)                      [512×16×16]

Decoder:
  Level 4: TranspConv + cat(skip4) + DoubleConv(512→256)
  Level 3: TranspConv + cat(skip3) + DoubleConv(256→128)
  Level 2: TranspConv + cat(skip2) + DoubleConv(128→64)
  Level 1: TranspConv + cat(skip1) + DoubleConv(64→32)

Output: Conv1×1(32→1) → Sigmoid → (B, 1, 256, 256)
```

**Total parameters**: ~7.7 million

---

## 📊 Training Configuration

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Batch Size | 8 |
| Epochs | 30 (max) |
| Loss Function | Dice Loss + BCE (α=0.5) |
| LR Scheduler | ReduceLROnPlateau (patience=3) |
| Early Stopping | patience=6 epochs |
| Data Split | 70% Train / 15% Val / 15% Test |
| Splitting | Patient-level (no slice leakage) |
| Sampling | Dataset-aware weighted sampler |

---

## 📈 Evaluation Metrics

| Metric | Formula |
|--------|---------|
| Dice (DSC) | 2TP / (2TP+FP+FN) |
| IoU (Jaccard) | TP / (TP+FP+FN) |
| Accuracy | (TP+TN) / (TP+TN+FP+FN) |
| Precision | TP / (TP+FP) |
| Recall | TP / (TP+FN) |
| F1 Score | 2·P·R / (P+R) |
| Sensitivity | TP / (TP+FN) |
| Specificity | TN / (TN+FP) |

---

## ⚠️ Common Errors & Solutions

| Error | Solution |
|-------|----------|
| `No DICOM series found` | Check that `.dcm` files exist in the patient subfolder |
| `Shape mismatch after resampling` | Ensure image & mask are from the same study |
| `highdicom not installed` | `pip install highdicom` |
| `pynrrd not installed` | `pip install pynrrd` |
| `CUDA out of memory` | Reduce `--batch_size` to 4 |
| `No preprocessed data found` | Run preprocessing scripts first |
| Lesion masks used as prostate | Verify PROSTATEx uses Seg-HiRes (not lesion CSV) |
| Patient overlap 3T / Dx | Use `check_patient_overlap()` in `nci_dx_preprocessing.py` |

---

## 🔬 Technology Stack

| Category | Library |
|----------|---------|
| Deep Learning | PyTorch ≥ 2.0 |
| Medical Imaging | NiBabel, SimpleITK, pydicom, pynrrd, highdicom |
| Image Processing | NumPy, OpenCV, Pillow, scikit-image |
| Evaluation | Scikit-learn, Pandas |
| Visualisation | Matplotlib |
| Web UI | Streamlit |
| Environment | Google Colab (GPU) / Local Python 3.9+ |

---

## 📝 Citation / References

If you use this project in academic work, please cite:

```
Ronneberger, O., Fischer, P., & Brox, T. (2015).
U-Net: Convolutional Networks for Biomedical Image Segmentation.
MICCAI 2015. https://doi.org/10.1007/978-3-319-24574-4_28
```

Dataset citations are available at their respective access URLs.

---

*MCA Academic AI/ML Project · Prostate MRI Segmentation · U-Net · PyTorch*

