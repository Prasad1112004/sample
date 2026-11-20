"""
=============================================================================
app.py  –  Streamlit Web Application
AI-Based Prostate MRI Segmentation Using Combined U-Net
=============================================================================

Run:
  streamlit run app.py

"""

import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prostate MRI Segmentation",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Project imports (graceful fallback if run without full project)
try:
    sys.path.insert(0, str(Path(__file__).parent))
    import torch
    from models.unet import build_unet
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False

import cv2
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Constants & Dataset Info
# ─────────────────────────────────────────────────────────────────────────────
DATASET_INFO = {
    "I2CVB": {
        "full_name"   : "Initiative for Collaborative Computer Vision Benchmarking",
        "scanner"     : "T2-weighted MRI",
        "approx_patients": 39,
        "approx_slices"  : 762,
        "format"      : "NIfTI / PNG slices (256x256)",
        "annotation"  : "Whole-gland segmentation",
        "access"      : "https://i2cvb.github.io/",
    },
    "MSD (Task05)": {
        "full_name"   : "Medical Segmentation Decathlon – Task05 Prostate",
        "scanner"     : "T2-weighted + ADC (multi-channel)",
        "approx_patients": 32,
        "approx_slices"  : 475,
        "format"      : "NIfTI / PNG slices (256x256)",
        "annotation"  : "PZ (label 1) + TZ (label 2) → merged whole-gland",
        "access"      : "http://medicaldecathlon.com/",
    },
    "NCI-ISBI 3T": {
        "full_name"   : "NCI-ISBI 2013 Challenge – 3 Tesla Scanner",
        "scanner"     : "3T T2-weighted MRI",
        "approx_patients": 40,
        "approx_slices"  : 555,
        "format"      : "DICOM + NRRD / PNG slices (256x256)",
        "annotation"  : "CG (1) + PZ (2) → merged whole-gland",
        "access"      : "https://wiki.cancerimagingarchive.net/",
    },
    "NCI-ISBI Dx": {
        "full_name"   : "NCI-ISBI 2013 Challenge – Diagnostic (1.5 Tesla)",
        "scanner"     : "1.5T T2-weighted MRI",
        "approx_patients": 39,
        "approx_slices"  : 483,
        "format"      : "DICOM + NRRD / PNG slices (256x256)",
        "annotation"  : "CG (1) + PZ (2) → merged whole-gland",
        "access"      : "https://wiki.cancerimagingarchive.net/",
    },
    "PROSTATEx": {
        "full_name"   : "PROSTATEx – PROSTATEx-Seg-HiRes",
        "scanner"     : "T2-weighted MRI",
        "approx_patients": 98,
        "approx_slices"  : 1627,
        "format"      : "DICOM + DICOM-SEG / PNG slices (256x256)",
        "annotation"  : "Whole-gland (Seg-HiRes) – NOT lesion annotations",
        "access"      : "https://wiki.cancerimagingarchive.net/",
    },
}

METRICS_ORDER = ["Dice", "IoU", "Accuracy", "Precision",
                 "Recall", "F1", "Sensitivity", "Specificity"]

DEFAULT_CHECKPOINT = "results/models/combined_prostate_unet.pth"
DEFAULT_METRICS_CSV = "results/metrics.csv"
TARGET_SIZE = (256, 256)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────────────────────────────────────
def sidebar():
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/10/"
                     "PyTorch_logo_icon.svg/1200px-PyTorch_logo_icon.svg.png",
                     width=60)
    st.sidebar.title("🔬 Prostate MRI Segmentation")
    page = st.sidebar.radio(
        "Navigate",
        ["🏠 Home",
         "📦 Dataset Information",
         "🧬 MRI Prediction",
         "📊 Model Evaluation",
         "📈 Performance Dashboard"],
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Tech Stack**: PyTorch · U-Net · Streamlit  \n"
        "**Datasets**: I2CVB · MSD · NCI-ISBI 3T/Dx · PROSTATEx  \n"
        "**Task**: Whole-prostate MRI segmentation"
    )
    return page


# ─────────────────────────────────────────────────────────────────────────────
# Page 1 – Home
# ─────────────────────────────────────────────────────────────────────────────
def page_home():
    st.title("🔬 AI-Based Prostate MRI Image Segmentation")
    st.subheader("Using a Combined U-Net Deep Learning Model with Five Datasets")

    st.markdown("""
---
### 📌 Project Overview
This project develops an automated whole-prostate segmentation system using the
**U-Net** deep learning architecture applied to prostate MRI images.  
A **single combined model** is trained on five publicly available datasets and
evaluated using eight clinical performance metrics.

---
### 🎯 Objectives
| # | Objective |
|---|-----------|
| 1 | Preprocess five heterogeneous prostate MRI datasets into a unified format |
| 2 | Train a single U-Net model on all combined data |
| 3 | Automatically segment the whole prostate gland from T2-weighted MRI |
| 4 | Evaluate with 8 metrics: DSC · IoU · Accuracy · Precision · Recall · F1 · Sensitivity · Specificity |
| 5 | Compare performance across all five datasets |

---
### 🏗️ U-Net Architecture
""")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Encoder Path**
- 4 stages of DoubleConv + MaxPool
- Channel progression: 1 → 32 → 64 → 128 → 256
- ReLU activation + Batch Normalisation

**Bottleneck**
- DoubleConv: 256 → 512 channels
- Captures high-level semantic features
""")
    with col2:
        st.markdown("""
**Decoder Path**
- 4 stages of TransposedConv + DoubleConv
- Skip connections from encoder
- Channel regression: 512 → 256 → 128 → 64 → 32

**Output**
- 1×1 Conv → 1 channel
- Sigmoid → binary prostate mask
- Input/Output: 256 × 256 × 1
""")

    st.markdown("""
---
### 📐 Architecture Diagram

```
Input (1×256×256)
│
├─ EncoderBlock1 ──────────────────────────────────────────────────┐
│   [Conv3×3→BN→ReLU] × 2 → MaxPool(2)  [ch: 1→32]               │ skip1
├─ EncoderBlock2 ──────────────────────────────────────────────────┐│
│   [Conv3×3→BN→ReLU] × 2 → MaxPool(2)  [ch: 32→64]              ││ skip2
├─ EncoderBlock3 ──────────────────────────────────────────────────┐││
│   [Conv3×3→BN→ReLU] × 2 → MaxPool(2)  [ch: 64→128]             │││ skip3
├─ EncoderBlock4 ──────────────────────────────────────────────────┐│││
│   [Conv3×3→BN→ReLU] × 2 → MaxPool(2)  [ch: 128→256]            ││││ skip4
│                                                                   ││││
├─ Bottleneck: [Conv3×3→BN→ReLU] × 2   [ch: 256→512]             ││││
│                                                                   ││││
├─ DecoderBlock4: TranspConv + concat(skip4) + DoubleConv ←────────┘│││
├─ DecoderBlock3: TranspConv + concat(skip3) + DoubleConv ←─────────┘││
├─ DecoderBlock2: TranspConv + concat(skip2) + DoubleConv ←──────────┘│
├─ DecoderBlock1: TranspConv + concat(skip1) + DoubleConv ←───────────┘
│
└─ Output Conv (1×1) → Sigmoid → Binary Mask (1×256×256)
```
""")


# ─────────────────────────────────────────────────────────────────────────────
# Page 2 – Dataset Information
# ─────────────────────────────────────────────────────────────────────────────
def page_datasets():
    st.title("📦 Dataset Information")
    st.markdown("Five publicly available prostate MRI datasets are combined "
                "into one unified training pipeline.")

    for name, info in DATASET_INFO.items():
        with st.expander(f"📁  {name}", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Full Name**: {info['full_name']}")
                st.markdown(f"**MRI Type** : {info['scanner']}")
                st.markdown(f"**Format**   : {info['format']}")
                st.markdown(f"**Annotation**: {info['annotation']}")
            with col2:
                st.metric("Approx. Patients",
                          info.get("approx_patients", "N/A"))
                st.metric("Approx. MRI Slices",
                          info.get("approx_slices", "N/A"))
                st.markdown(f"[Access Dataset]({info['access']})")

    st.markdown("---")
    st.subheader("⚙️ Preprocessing Summary")
    st.markdown("""
| Step | Operation |
|------|-----------|
| 1 | Load raw MRI volumes (NIfTI / DICOM / NRRD) |
| 2 | Select T2-weighted sequence |
| 3 | Resample mask to image grid (nearest-neighbour) |
| 4 | Merge zone-specific labels → binary mask (0=BG, 1=Prostate) |
| 5 | Clip intensities (1st–99th percentile) → normalise to [0, 1] |
| 6 | Resize 2D slices to 256 × 256 |
| 7 | Filter slices with ≥ 50 prostate pixels |
| 8 | Save as `.npy` files |
| 9 | Generate verification PNG for each patient |

**Dataset split**: 70% Train · 15% Validation · 15% Test  
**Splitting strategy**: Patient-level (no slice leakage across subsets)
""")


# ─────────────────────────────────────────────────────────────────────────────
# Page 3 – MRI Prediction
# ─────────────────────────────────────────────────────────────────────────────
def _load_model(checkpoint_path: str, device):
    """Load the saved U-Net checkpoint."""
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model = build_unet(device=str(device))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def _preprocess_uploaded_image(uploaded_file) -> np.ndarray | None:
    """
    Accept PNG / JPG / NPY / NIfTI uploads.
    Returns (H=256, W=256) float32 normalised array or None on failure.
    """
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".npy"):
            arr = np.load(io.BytesIO(uploaded_file.read())).astype(np.float32)
            if arr.ndim == 3:
                arr = arr[arr.shape[0] // 2]   # take middle slice
        elif name.endswith((".nii", ".nii.gz")):
            if not NIBABEL_AVAILABLE:
                st.error("NiBabel not installed — cannot read NIfTI.")
                return None
            with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
                f.write(uploaded_file.read())
                tmp = f.name
            vol = nib.load(tmp).get_fdata().astype(np.float32)
            arr = vol[:, :, vol.shape[2] // 2]   # middle axial slice
        else:   # PNG / JPG
            img  = Image.open(uploaded_file).convert("L")   # greyscale
            arr  = np.array(img, dtype=np.float32)

        # Resize to 256×256
        arr = cv2.resize(arr, (256, 256), interpolation=cv2.INTER_LINEAR)

        # Normalise to [0, 1]
        lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
        if hi > lo:
            arr = np.clip(arr, lo, hi)
            arr = (arr - lo) / (hi - lo)
        return arr.astype(np.float32)

    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None


def page_prediction():
    st.title("🧬 MRI Prediction")
    st.markdown(
        "> **Supported formats**: `.npy` (preprocessed slice) · PNG · JPG · "
        "`.nii` / `.nii.gz` (NIfTI).  \n"
        "> The image must be a **T2-weighted prostate MRI** slice.  \n"
        "> Non-prostate MRI images will produce meaningless predictions."
    )

    # Model loading
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Model Settings")
    ckpt_path = st.sidebar.text_input("Checkpoint path", DEFAULT_CHECKPOINT)
    threshold = st.sidebar.slider("Prediction threshold", 0.1, 0.9, 0.5, 0.05)

    uploaded = st.file_uploader(
        "Upload a prostate MRI image",
        type=["npy", "png", "jpg", "jpeg", "nii", "gz"],
    )

    if uploaded is None:
        st.info("👆 Upload an MRI image to begin.")
        return

    # Preprocess
    with st.spinner("Preprocessing image …"):
        img_np = _preprocess_uploaded_image(uploaded)

    if img_np is None:
        return

    # Inference
    if not TORCH_AVAILABLE:
        st.error("PyTorch is not installed – cannot run inference.")
        return
    if not Path(ckpt_path).exists():
        st.warning(f"Checkpoint not found at `{ckpt_path}`. "
                   "Please train the model first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with st.spinner(f"Running U-Net inference on {device} …"):
        model, ckpt_meta = _load_model(ckpt_path, device)
        img_t  = torch.from_numpy(img_np[np.newaxis, np.newaxis]).to(device)
        with torch.no_grad():
            logits = model(img_t)
            probs  = torch.sigmoid(logits)[0, 0].cpu().numpy()
            pred   = (probs > threshold).astype(np.uint8)

    # Display
    st.success("✅ Segmentation complete!")
    st.markdown(f"*Trained at epoch {ckpt_meta.get('epoch', '?')} "
                f"| Val Dice: {ckpt_meta.get('val_dice', 0):.4f}*")

    col1, col2, col3 = st.columns(3)

    def _show(ax_img, title):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(ax_img, cmap="gray" if ax_img.ndim == 2 else None)
        ax.set_title(title);  ax.axis("off")
        return fig

    overlay = np.stack([img_np, img_np, img_np], -1)
    overlay[pred == 1, 0] = 1.0
    overlay[pred == 1, 1] = 0.3
    overlay[pred == 1, 2] = 0.0

    with col1:
        st.pyplot(_show(img_np, "Original MRI (T2w)"))
    with col2:
        st.pyplot(_show(pred, "Predicted Prostate Mask"))
    with col3:
        st.pyplot(_show(overlay, "Segmentation Overlay"))
        
    plt.close("all")

    # Probability map
    with st.expander("🔥 Probability Heatmap"):
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(probs, cmap="jet", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label="Probability")
        ax.set_title("Prostate Segmentation Probability")
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)

    # Quick metrics (prediction only – no GT available)
    prostate_px = int(pred.sum())
    total_px    = int(pred.size)
    coverage    = prostate_px / total_px * 100
    st.info(f"**Prostate pixels**: {prostate_px:,} / {total_px:,} "
            f"({coverage:.1f}% of image area)")


# ─────────────────────────────────────────────────────────────────────────────
# Page 4 – Model Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def page_evaluation():
    st.title("📊 Model Evaluation")
    st.markdown("Metrics calculated on the **held-out test set** "
                "(patient-level split, never seen during training).")

    if not Path(DEFAULT_METRICS_CSV).exists():
        st.warning(
            f"`{DEFAULT_METRICS_CSV}` not found.  \n"
            "Run `python evaluation/evaluate.py` after training to generate metrics."
        )
        _show_metric_explanations()
        return

    df = pd.read_csv(DEFAULT_METRICS_CSV)

    # Overall metrics
    st.subheader("🌐 Overall Performance")
    overall = {m: df[m].mean() for m in METRICS_ORDER}
    cols = st.columns(4)
    for i, (m, v) in enumerate(overall.items()):
        cols[i % 4].metric(m, f"{v:.4f}")

    st.markdown("---")

    # Per-dataset table
    st.subheader("🗂️ Per-Dataset Performance")
    ds_table = (
        df.groupby("dataset")[METRICS_ORDER]
        .agg(["mean", "std"])
    )
    # Flatten column names
    ds_table.columns = [f"{m}_{s}" for m, s in ds_table.columns]
    st.dataframe(ds_table.style.format("{:.4f}"), use_container_width=True)

    # Patient-level scatter
    st.subheader("🔵 Patient-Level Dice Distribution")
    fig, ax = plt.subplots(figsize=(10, 4))
    for ds in df["dataset"].unique():
        sub = df[df["dataset"] == ds]["Dice"]
        ax.scatter(range(len(sub)), sub.values, label=ds, s=20, alpha=0.7)
    ax.axhline(df["Dice"].mean(), color="red", linestyle="--",
               label=f"Overall Mean = {df['Dice'].mean():.3f}")
    ax.set_xlabel("Patient index");  ax.set_ylabel("Dice Score")
    ax.set_title("Patient-Level Dice Scores by Dataset")
    ax.legend(fontsize=8);  ax.grid(alpha=0.3)
    st.pyplot(fig);  plt.close(fig)

    _show_metric_explanations()


def _show_metric_explanations():
    st.markdown("---")
    st.subheader("📖 Metric Definitions")
    st.markdown("""
| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Dice (DSC)** | 2TP / (2TP+FP+FN) | Overlap similarity (0–1, higher=better) |
| **IoU** | TP / (TP+FP+FN) | Jaccard index (0–1) |
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Overall pixel correctness |
| **Precision** | TP / (TP+FP) | Fraction of predicted prostate that is correct |
| **Recall** | TP / (TP+FN) | Fraction of actual prostate detected |
| **F1 Score** | 2·P·R / (P+R) | Harmonic mean of Precision & Recall |
| **Sensitivity** | TP / (TP+FN) | Same as Recall (true positive rate) |
| **Specificity** | TN / (TN+FP) | True negative rate (background accuracy) |

**Notation**: TP=True Positive, TN=True Negative, FP=False Positive, FN=False Negative
""")


# ─────────────────────────────────────────────────────────────────────────────
# Page 5 – Performance Dashboard
# ─────────────────────────────────────────────────────────────────────────────
def page_dashboard():
    st.title("📈 Performance Dashboard")

    graphs_dir = Path("results/graphs")
    history_path = Path("results/training_history.json")

    # Training curves
    st.subheader("📉 Training & Validation Curves")
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        epochs = range(1, len(history["train_loss"]) + 1)

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            ax.plot(epochs, history["train_loss"], label="Train", color="steelblue")
            ax.plot(epochs, history["val_loss"],   label="Val",   color="coral")
            ax.set_title("Loss");  ax.legend();  ax.set_xlabel("Epoch")
            st.pyplot(fig);  plt.close(fig)
        with col2:
            fig, ax = plt.subplots()
            ax.plot(epochs, history["train_dice"], label="Train", color="steelblue")
            ax.plot(epochs, history["val_dice"],   label="Val",   color="coral")
            ax.set_title("Dice Score");  ax.legend();  ax.set_xlabel("Epoch")
            st.pyplot(fig);  plt.close(fig)
    else:
        st.info("Training history not found. Run training first.")

    # Saved graphs
    graph_files = {
        "Dataset Comparison"        : "dataset_comparison.png",
        "8 Metrics (Overall)"       : "eight_metrics_overall.png",
        "Precision & Recall"        : "precision_recall_comparison.png",
        "Loss & Dice Curves"        : "loss_dice_curves.png",
    }

    st.markdown("---")
    st.subheader("📊 Generated Performance Graphs")

    for title, fname in graph_files.items():
        fpath = graphs_dir / fname
        if fpath.exists():
            st.markdown(f"**{title}**")
            st.image(str(fpath), use_column_width=True)
        else:
            st.caption(f"*{title}* – not yet generated (run evaluation first).")

    # Predictions gallery
    pred_dir = Path("results/predictions")
    if pred_dir.exists():
        pred_imgs = sorted(pred_dir.glob("*.png"))[:6]   # show up to 6
        if pred_imgs:
            st.markdown("---")
            st.subheader("🖼️ Sample Predictions")
            cols = st.columns(3)
            for i, p in enumerate(pred_imgs):
                cols[i % 3].image(str(p), caption=p.stem, use_column_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main Router
# ─────────────────────────────────────────────────────────────────────────────
def main():
    page = sidebar()

    if   page == "🏠 Home":                 page_home()
    elif page == "📦 Dataset Information":  page_datasets()
    elif page == "🧬 MRI Prediction":       page_prediction()
    elif page == "📊 Model Evaluation":     page_evaluation()
    elif page == "📈 Performance Dashboard": page_dashboard()


if __name__ == "__main__":
    main()

