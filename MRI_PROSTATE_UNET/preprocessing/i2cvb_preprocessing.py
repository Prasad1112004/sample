"""
=============================================================================
i2cvb_preprocessing.py
Dataset : I2CVB (Initiative for Collaborative Computer Vision Benchmarking)
Format  : NIfTI (.nii / .nii.gz)
MRI type: T2-weighted whole-gland segmentation
Labels  : Whole prostate gland  → binary mask (0 background, 1 prostate)
=============================================================================

Expected dataset folder layout
--------------------------------
datasets/I2CVB/
└── <patient_id>/
    ├── T2W.nii.gz          # or T2.nii.gz / t2w.nii.gz
    └── prostate.nii.gz     # or label.nii.gz / segmentation.nii.gz

The I2CVB dataset page: https://i2cvb.github.io/
Access : Free academic download (registration required).

Outputs
-------
preprocessed/I2CVB/<patient_id>/
    image_slice_<N>.npy   (float32, 256×256, normalised)
    mask_slice_<N>.npy    (uint8,   256×256, binary 0/1)
    meta.json             (spacing, direction, origin, n_slices)
"""

import os
import json
import glob
import logging
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import SimpleITK as sitk
import cv2
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  [I2CVB]  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
TARGET_SIZE = (256, 256)          # (H, W)
CLIP_PERCENTILE = (1, 99)         # intensity clipping for normalisation
MIN_PROSTATE_PIXELS = 50          # discard near-empty slices


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _find_file(folder: Path, patterns: list[str]) -> Path | None:
    """Return the first file that matches any glob pattern, or None."""
    for pat in patterns:
        hits = list(folder.glob(pat))
        if hits:
            return hits[0]
    return None


def _load_nifti_as_array(path: Path) -> tuple[np.ndarray, object]:
    """Load a NIfTI file with NiBabel and return (data_float32, affine)."""
    img = nib.load(str(path))
    data = np.array(img.get_fdata(), dtype=np.float32)
    return data, img.affine


def _resample_mask_to_image(mask_path: Path,
                             image_path: Path) -> np.ndarray:
    """
    Resample the mask volume to exactly match the image voxel grid using
    SimpleITK nearest-neighbour interpolation, then return as uint8 array.
    """
    ref_img = sitk.ReadImage(str(image_path), sitk.sitkFloat32)
    mask_img = sitk.ReadImage(str(mask_path), sitk.sitkUInt8)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_img)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampled = resampler.Execute(mask_img)

    arr = sitk.GetArrayFromImage(resampled)   # shape: (D, H, W)
    return arr.astype(np.uint8)


def _normalize_intensity(volume: np.ndarray) -> np.ndarray:
    """Clip to percentile range then rescale to [0, 1]."""
    lo, hi = np.percentile(volume[volume > 0], CLIP_PERCENTILE) \
        if volume.max() > 0 else (volume.min(), volume.max())
    volume = np.clip(volume, lo, hi)
    rng = hi - lo
    if rng > 0:
        volume = (volume - lo) / rng
    return volume.astype(np.float32)


def _resize_slice(arr: np.ndarray, size: tuple[int, int],
                  interpolation=cv2.INTER_LINEAR) -> np.ndarray:
    """Resize a single 2D slice; preserve dtype."""
    return cv2.resize(arr.astype(np.float32), (size[1], size[0]),
                      interpolation=interpolation).astype(arr.dtype)


def _binarize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert any non-zero label to 1 (whole-prostate)."""
    return (mask > 0).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Core preprocessing function
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_patient(patient_dir: Path,
                        out_dir: Path,
                        visualize: bool = True) -> dict:
    """
    Preprocess one I2CVB patient.

    Parameters
    ----------
    patient_dir : Path   Root folder for this patient.
    out_dir     : Path   Destination for standardised slices.
    visualize   : bool   Save a PNG verification strip.

    Returns
    -------
    dict  Metadata (patient_id, n_slices, spacing, …).
    """
    patient_id = patient_dir.name
    logger.info("Processing patient %s", patient_id)

    # ── 1. Locate image & mask files ──────────────────────────────────────
    img_patterns = ["*T2W*.nii*", "*T2*.nii*", "*t2w*.nii*", "*image*.nii*"]
    msk_patterns = ["*prostate*.nii*", "*label*.nii*", "*segmentation*.nii*",
                    "*mask*.nii*", "*Prostate*.nii*", "*Label*.nii*"]

    img_path = _find_file(patient_dir, img_patterns)
    msk_path = _find_file(patient_dir, msk_patterns)

    if img_path is None or msk_path is None:
        logger.warning("  ✗ Missing image or mask for %s — skipping.", patient_id)
        return {}

    logger.info("  Image : %s", img_path.name)
    logger.info("  Mask  : %s", msk_path.name)

    # ── 2. Load image (NiBabel) ───────────────────────────────────────────
    image_vol, affine = _load_nifti_as_array(img_path)   # (H, W, D) or (D, H, W)

    # ── 3. Resample mask to match image grid (SimpleITK) ──────────────────
    mask_vol = _resample_mask_to_image(msk_path, img_path)

    # ── 4. Align array orientation ────────────────────────────────────────
    # NiBabel returns (X, Y, Z); we want axial slices → transpose to (Z, X, Y)
    if image_vol.ndim == 3:
        # Heuristic: smallest dimension is likely slice direction
        slice_axis = int(np.argmin(image_vol.shape))
        image_vol = np.moveaxis(image_vol, slice_axis, 0)  # → (D, H, W)
        mask_vol  = np.moveaxis(mask_vol,  slice_axis, 0)

    assert image_vol.shape == mask_vol.shape, (
        f"Shape mismatch after resampling: {image_vol.shape} vs {mask_vol.shape}")

    # ── 5. Binarize mask ──────────────────────────────────────────────────
    mask_vol = _binarize_mask(mask_vol)

    # ── 6. Intensity normalisation (volume-level) ─────────────────────────
    image_vol = _normalize_intensity(image_vol)

    # ── 7. Select slices with sufficient prostate content ─────────────────
    n_slices_total = image_vol.shape[0]
    kept_slices = []
    for z in range(n_slices_total):
        if mask_vol[z].sum() >= MIN_PROSTATE_PIXELS:
            kept_slices.append(z)

    # Also include adjacent slices (context) – optional but helpful
    context_slices = set()
    for z in kept_slices:
        for dz in [-1, 0, 1]:
            idx = z + dz
            if 0 <= idx < n_slices_total:
                context_slices.add(idx)
    all_slices = sorted(context_slices)

    if not all_slices:
        logger.warning("  ✗ No prostate slices found for %s — skipping.", patient_id)
        return {}

    # ── 8. Resize slices and save ─────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    vis_data = []   # for verification PNG

    for z in all_slices:
        img_slice  = _resize_slice(image_vol[z], TARGET_SIZE, cv2.INTER_LINEAR)
        mask_slice = _resize_slice(mask_vol[z].astype(np.float32),
                                   TARGET_SIZE, cv2.INTER_NEAREST).astype(np.uint8)

        np.save(str(out_dir / f"image_slice_{z:04d}.npy"), img_slice)
        np.save(str(out_dir / f"mask_slice_{z:04d}.npy"),  mask_slice)
        saved += 1

        if visualize and len(vis_data) < 4:
            vis_data.append((img_slice, mask_slice, z))

    # ── 9. Save metadata ───────────────────────────────────────────────────
    # Extract voxel spacing from SimpleITK
    sitk_img = sitk.ReadImage(str(img_path))
    spacing = list(sitk_img.GetSpacing())   # (x, y, z) mm

    meta = {
        "patient_id"    : patient_id,
        "dataset"       : "I2CVB",
        "image_path"    : str(img_path),
        "mask_path"     : str(msk_path),
        "original_shape": list(image_vol.shape),
        "n_slices_total": n_slices_total,
        "n_slices_saved": saved,
        "spacing_mm"    : spacing,
        "target_size"   : list(TARGET_SIZE),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # ── 10. Verification visualisation ────────────────────────────────────
    if visualize and vis_data:
        _save_verification_png(vis_data, out_dir, patient_id)

    logger.info("  ✓ Saved %d slices → %s", saved, out_dir)
    return meta


def _save_verification_png(vis_data: list, out_dir: Path, patient_id: str):
    """Save a 4-column verification strip: image | mask | overlay."""
    n = len(vis_data)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, (img, msk, z) in enumerate(vis_data):
        overlay = np.stack([img, img, img], axis=-1)
        overlay[msk == 1, 0] = 1.0   # red channel for prostate
        overlay[msk == 1, 1] = 0.0
        overlay[msk == 1, 2] = 0.0

        axes[row, 0].imshow(img, cmap="gray")
        axes[row, 0].set_title(f"T2W slice z={z}")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(msk, cmap="gray")
        axes[row, 1].set_title("Binary mask")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title("Overlay")
        axes[row, 2].axis("off")

    fig.suptitle(f"I2CVB – {patient_id}", fontsize=12)
    plt.tight_layout()
    plt.savefig(str(out_dir / "verification.png"), dpi=100)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Batch runner
# ─────────────────────────────────────────────────────────────────────────────
def run_all(dataset_root: str,
            output_root: str,
            visualize: bool = True) -> list[dict]:
    """
    Iterate over all patient sub-folders in dataset_root and preprocess each.

    Parameters
    ----------
    dataset_root : str   Path to datasets/I2CVB/
    output_root  : str   Path to preprocessed/I2CVB/

    Returns
    -------
    list of metadata dicts for all successfully processed patients.
    """
    root = Path(dataset_root)
    out  = Path(output_root)
    all_meta = []

    patient_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not patient_dirs:
        logger.error("No patient directories found in %s", root)
        return all_meta

    logger.info("Found %d patient directories.", len(patient_dirs))

    for pdir in patient_dirs:
        meta = preprocess_patient(pdir, out / pdir.name, visualize=visualize)
        if meta:
            all_meta.append(meta)

    logger.info("I2CVB: processed %d / %d patients.",
                len(all_meta), len(patient_dirs))

    # Summary CSV
    import pandas as pd
    df = pd.DataFrame(all_meta)
    df.to_csv(out / "i2cvb_summary.csv", index=False)
    logger.info("Summary saved → %s", out / "i2cvb_summary.csv")

    return all_meta


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="I2CVB Dataset Preprocessor")
    parser.add_argument("--dataset_root", type=str,
                        default="datasets/I2CVB",
                        help="Path to raw I2CVB dataset folder")
    parser.add_argument("--output_root", type=str,
                        default="preprocessed/I2CVB",
                        help="Output folder for standardised slices")
    parser.add_argument("--no_vis", action="store_true",
                        help="Disable verification PNG generation")
    args = parser.parse_args()

    all_meta = run_all(args.dataset_root, args.output_root,
                       visualize=not args.no_vis)
    print(f"\nI2CVB preprocessing complete. Patients processed: {len(all_meta)}")

