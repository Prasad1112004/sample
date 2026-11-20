"""
=============================================================================
preprocess_all.py
Universal Preprocessor for all Five Prostate MRI Datasets
Processes raw image/mask slices from datasets/ into preprocessed/
=============================================================================
"""

import os
import json
import logging
import argparse
from pathlib import Path
import re

import numpy as np
import cv2
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  [PREPROC]  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

TARGET_SIZE = (256, 256)
CLIP_PERCENTILE = (1, 99)
DATASETS = ["I2CVB", "MSD", "NCI_ISBI_3T", "NCI_ISBI_Dx", "PROSTATEx"]


def extract_slice_num(filename: str) -> int:
    """Extract integer slice index from filename like '12.png' or 'slice_012.png'."""
    nums = re.findall(r'\d+', filename)
    return int(nums[-1]) if nums else 0


def normalize_intensity(img: np.ndarray) -> np.ndarray:
    """Percentile clipping and min-max normalization to [0, 1]."""
    img_f = img.astype(np.float32)
    fg = img_f[img_f > 0]
    if fg.size == 0:
        return img_f
    lo, hi = np.percentile(fg, CLIP_PERCENTILE)
    img_f = np.clip(img_f, lo, hi)
    if hi > lo:
        img_f = (img_f - lo) / (hi - lo)
    return img_f.astype(np.float32)


def preprocess_patient(dataset_name: str,
                       patient_id: str,
                       img_dir: Path,
                       mask_dir: Path,
                       out_dir: Path,
                       visualize: bool = True) -> dict:
    """Process all slices for one patient."""
    img_files = sorted(img_dir.glob("*.png"), key=lambda f: extract_slice_num(f.name))
    if not img_files:
        return {}

    out_dir.mkdir(parents=True, exist_ok=True)
    vis_data = []
    saved_count = 0

    for img_path in img_files:
        mask_path = mask_dir / img_path.name
        if not mask_path.exists():
            continue

        # Load image & mask
        img_raw = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if img_raw is None or mask_raw is None:
            continue

        # Ensure spatial alignment
        if img_raw.shape != mask_raw.shape:
            mask_raw = cv2.resize(mask_raw, (img_raw.shape[1], img_raw.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)

        # Normalize image
        img_norm = normalize_intensity(img_raw)

        # Binarize mask: 0=background, 1=prostate
        mask_bin = (mask_raw > 127).astype(np.uint8)

        # Resize to standard 256x256
        img_res = cv2.resize(img_norm, TARGET_SIZE, interpolation=cv2.INTER_LINEAR).astype(np.float32)
        mask_res = cv2.resize(mask_bin, TARGET_SIZE, interpolation=cv2.INTER_NEAREST).astype(np.uint8)

        z = extract_slice_num(img_path.name)
        np.save(str(out_dir / f"image_slice_{z:04d}.npy"), img_res)
        np.save(str(out_dir / f"mask_slice_{z:04d}.npy"),  mask_res)
        saved_count += 1

        if visualize and len(vis_data) < 3 and mask_res.sum() > 50:
            vis_data.append((img_res, mask_res, z))

    if saved_count == 0:
        return {}

    # Save verification PNG
    if visualize and vis_data:
        _save_verification_png(vis_data, out_dir, f"{dataset_name} - {patient_id}")

    meta = {
        "dataset": dataset_name,
        "patient_id": patient_id,
        "original_shape": [int(img_raw.shape[0]), int(img_raw.shape[1])],
        "target_size": list(TARGET_SIZE),
        "n_slices_saved": saved_count
    }

    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return meta


def _save_verification_png(vis_data, out_dir, title):
    n = len(vis_data)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for r, (img, msk, z) in enumerate(vis_data):
        overlay = np.stack([img, img, img], -1)
        overlay[msk == 1, 0] = 1.0
        overlay[msk == 1, 1:] = 0.0

        axes[r, 0].imshow(img, cmap="gray")
        axes[r, 0].set_title(f"T2w slice z={z}")
        axes[r, 0].axis("off")

        axes[r, 1].imshow(msk, cmap="gray")
        axes[r, 1].set_title("Binary mask")
        axes[r, 1].axis("off")

        axes[r, 2].imshow(overlay)
        axes[r, 2].set_title("Overlay")
        axes[r, 2].axis("off")

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(str(out_dir / "verification.png"), dpi=100)
    plt.close(fig)


def preprocess_dataset(dataset_root: Path, output_root: Path, dataset_name: str) -> list[dict]:
    """Preprocess all patients in one dataset directory."""
    img_root = dataset_root / dataset_name / "Image"
    mask_root = dataset_root / dataset_name / "Mask"
    out_ds_root = output_root / dataset_name

    if not img_root.exists() or not mask_root.exists():
        logger.warning("Images or Masks not found for %s", dataset_name)
        return []

    patient_dirs = sorted([d for d in img_root.iterdir() if d.is_dir()])
    logger.info("Dataset [%s]: Found %d patients", dataset_name, len(patient_dirs))

    metas = []
    for pdir in patient_dirs:
        mask_pdir = mask_root / pdir.name
        if not mask_pdir.exists():
            continue
        meta = preprocess_patient(
            dataset_name=dataset_name,
            patient_id=pdir.name,
            img_dir=pdir,
            mask_dir=mask_pdir,
            out_dir=out_ds_root / pdir.name,
            visualize=True
        )
        if meta:
            metas.append(meta)

    logger.info("Dataset [%s]: Successfully processed %d / %d patients",
                dataset_name, len(metas), len(patient_dirs))
    return metas


def run_all(dataset_root: str = "datasets", output_root: str = "preprocessed"):
    ds_root = Path(dataset_root)
    out_root = Path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    total_slices = 0
    total_patients = 0

    for ds_name in DATASETS:
        metas = preprocess_dataset(ds_root, out_root, ds_name)
        total_patients += len(metas)
        total_slices += sum(m["n_slices_saved"] for m in metas)

    logger.info("=" * 60)
    logger.info("PREPROCESSING COMPLETE!")
    logger.info("Total Patients: %d", total_patients)
    logger.info("Total Slices  : %d", total_slices)
    logger.info("Output Dir    : %s", out_root.resolve())
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="datasets", help="Root folder of raw datasets")
    parser.add_argument("--output_root", default="preprocessed", help="Root folder for preprocessed output")
    args = parser.parse_args()
    run_all(args.dataset_root, args.output_root)

