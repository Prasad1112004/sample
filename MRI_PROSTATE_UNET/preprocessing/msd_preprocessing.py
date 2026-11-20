"""
=============================================================================
msd_preprocessing.py
Dataset : Medical Segmentation Decathlon – Task05_Prostate
Format  : NIfTI (.nii.gz)
MRI type: Multi-channel – channel 0 = T2w, channel 1 = ADC
Labels  : 1 = Peripheral Zone (PZ), 2 = Transition Zone (TZ)
          → merged → binary whole-prostate mask
=============================================================================

Expected dataset folder layout
--------------------------------
datasets/MSD/Task05_Prostate/
├── dataset.json
├── imagesTr/
│   └── prostate_<NNN>_0000.nii.gz   (T2w channel = _0000)
│   └── prostate_<NNN>_0001.nii.gz   (ADC  channel = _0001)
└── labelsTr/
    └── prostate_<NNN>.nii.gz

Download: http://medicaldecathlon.com/

Outputs
-------
preprocessed/MSD/<case_id>/
    image_slice_<N>.npy   (float32, 256×256, normalised T2w)
    mask_slice_<N>.npy    (uint8,   256×256, binary PZ+TZ → 1)
    meta.json
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
                    format="%(asctime)s  [MSD]  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

TARGET_SIZE = (256, 256)
CLIP_PERCENTILE = (1, 99)
MIN_PROSTATE_PIXELS = 50
T2W_CHANNEL_SUFFIX = "_0000"   # T2w is always channel 0000 in MSD convention


def _load_nifti(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    return np.array(img.get_fdata(), dtype=np.float32)


def _resample_to_reference(moving_path: Path,
                            fixed_path: Path,
                            interpolator=sitk.sitkNearestNeighbor) -> np.ndarray:
    """Resample moving image to match fixed image grid."""
    fixed  = sitk.ReadImage(str(fixed_path),  sitk.sitkFloat32)
    moving = sitk.ReadImage(str(moving_path), sitk.sitkFloat32)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0)
    out = resampler.Execute(moving)
    return sitk.GetArrayFromImage(out)   # (D, H, W)


def _normalize(volume: np.ndarray) -> np.ndarray:
    fg = volume[volume > 0]
    if fg.size == 0:
        return volume
    lo, hi = np.percentile(fg, CLIP_PERCENTILE)
    volume = np.clip(volume, lo, hi)
    if hi > lo:
        volume = (volume - lo) / (hi - lo)
    return volume.astype(np.float32)


def _resize_2d(arr: np.ndarray, size: tuple, interp=cv2.INTER_LINEAR) -> np.ndarray:
    return cv2.resize(arr.astype(np.float32), (size[1], size[0]),
                      interpolation=interp).astype(arr.dtype)


def _binarize(mask: np.ndarray) -> np.ndarray:
    """Merge label 1 (PZ) and label 2 (TZ) → 1 (whole prostate)."""
    return ((mask == 1) | (mask == 2)).astype(np.uint8)


def preprocess_case(case_id: str,
                    images_dir: Path,
                    labels_dir: Path,
                    out_dir: Path,
                    visualize: bool = True) -> dict:
    """
    Preprocess one MSD Task05 case.

    Parameters
    ----------
    case_id    : e.g. "prostate_00"
    images_dir : imagesTr/
    labels_dir : labelsTr/
    out_dir    : destination
    visualize  : save verification PNG
    """
    logger.info("Processing case %s", case_id)

    # ── 1. Locate T2w image ────────────────────────────────────────────────
    t2w_path  = images_dir / f"{case_id}{T2W_CHANNEL_SUFFIX}.nii.gz"
    mask_path = labels_dir / f"{case_id}.nii.gz"

    if not t2w_path.exists():
        logger.warning("  ✗ T2w image not found: %s", t2w_path)
        return {}
    if not mask_path.exists():
        logger.warning("  ✗ Label not found: %s", mask_path)
        return {}

    # ── 2. Load T2w volume ────────────────────────────────────────────────
    t2w_nib = nib.load(str(t2w_path))
    t2w_vol  = np.array(t2w_nib.get_fdata(), dtype=np.float32)  # (H,W,D)

    # ── 3. Load & resample mask to T2w grid ───────────────────────────────
    mask_vol = _resample_to_reference(mask_path, t2w_path,
                                      sitk.sitkNearestNeighbor)

    # MSD volumes are already (D, H, W) from SimpleITK
    # NiBabel returns (H, W, D) → transpose
    if t2w_vol.shape != mask_vol.shape:
        # Transpose T2w so axes align with mask from SimpleITK
        t2w_vol = np.transpose(t2w_vol, (2, 0, 1))   # (D, H, W)

    assert t2w_vol.shape == mask_vol.shape, \
        f"Shape mismatch: {t2w_vol.shape} vs {mask_vol.shape}"

    # ── 4. Binarize mask ──────────────────────────────────────────────────
    mask_vol = _binarize(mask_vol.astype(np.int32))

    # ── 5. Normalise intensity ────────────────────────────────────────────
    t2w_vol = _normalize(t2w_vol)

    # ── 6. Select prostate slices ─────────────────────────────────────────
    prostate_slices = [z for z in range(mask_vol.shape[0])
                       if mask_vol[z].sum() >= MIN_PROSTATE_PIXELS]
    context = set()
    for z in prostate_slices:
        for dz in [-1, 0, 1]:
            idx = z + dz
            if 0 <= idx < mask_vol.shape[0]:
                context.add(idx)
    all_slices = sorted(context)

    if not all_slices:
        logger.warning("  ✗ No prostate slices for %s — skipping.", case_id)
        return {}

    # ── 7. Resize & save ──────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_data = []

    for z in all_slices:
        img_s  = _resize_2d(t2w_vol[z],             TARGET_SIZE, cv2.INTER_LINEAR)
        msk_s  = _resize_2d(mask_vol[z].astype(np.float32),
                             TARGET_SIZE, cv2.INTER_NEAREST).astype(np.uint8)
        np.save(str(out_dir / f"image_slice_{z:04d}.npy"), img_s)
        np.save(str(out_dir / f"mask_slice_{z:04d}.npy"),  msk_s)
        if visualize and len(vis_data) < 4:
            vis_data.append((img_s, msk_s, z))

    # ── 8. Metadata ───────────────────────────────────────────────────────
    sitk_img = sitk.ReadImage(str(t2w_path))
    meta = {
        "patient_id"    : case_id,
        "dataset"       : "MSD",
        "t2w_path"      : str(t2w_path),
        "mask_path"     : str(mask_path),
        "original_shape": list(t2w_vol.shape),
        "n_slices_saved": len(all_slices),
        "spacing_mm"    : list(sitk_img.GetSpacing()),
        "target_size"   : list(TARGET_SIZE),
        "labels_merged" : "PZ(1) + TZ(2) → 1",
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    if visualize and vis_data:
        _save_verification_png(vis_data, out_dir, case_id)

    logger.info("  ✓ Saved %d slices → %s", len(all_slices), out_dir)
    return meta


def _save_verification_png(vis_data, out_dir, title):
    n = len(vis_data)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    for r, (img, msk, z) in enumerate(vis_data):
        ov = np.stack([img, img, img], -1)
        ov[msk == 1, 0] = 1.0; ov[msk == 1, 1:] = 0.0
        axes[r, 0].imshow(img, cmap="gray");  axes[r, 0].set_title(f"T2w z={z}");   axes[r, 0].axis("off")
        axes[r, 1].imshow(msk, cmap="gray");  axes[r, 1].set_title("Binary mask");   axes[r, 1].axis("off")
        axes[r, 2].imshow(ov);                axes[r, 2].set_title("Overlay");        axes[r, 2].axis("off")
    fig.suptitle(f"MSD Task05 – {title}")
    plt.tight_layout()
    plt.savefig(str(out_dir / "verification.png"), dpi=100)
    plt.close(fig)


def run_all(dataset_root: str,
            output_root: str,
            visualize: bool = True) -> list[dict]:
    root = Path(dataset_root)
    images_dir = root / "imagesTr"
    labels_dir = root / "labelsTr"

    if not images_dir.exists():
        logger.error("imagesTr not found in %s", root)
        return []

    # Collect unique case IDs
    t2w_files = sorted(images_dir.glob(f"*{T2W_CHANNEL_SUFFIX}.nii.gz"))
    case_ids  = [f.name.replace(f"{T2W_CHANNEL_SUFFIX}.nii.gz", "") for f in t2w_files]

    logger.info("Found %d T2w images.", len(case_ids))
    all_meta = []
    out = Path(output_root)

    for cid in case_ids:
        meta = preprocess_case(cid, images_dir, labels_dir,
                               out / cid, visualize=visualize)
        if meta:
            all_meta.append(meta)

    import pandas as pd
    df = pd.DataFrame(all_meta)
    df.to_csv(out / "msd_summary.csv", index=False)
    logger.info("MSD: processed %d / %d cases.", len(all_meta), len(case_ids))
    return all_meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MSD Task05 Prostate Preprocessor")
    parser.add_argument("--dataset_root", default="datasets/MSD/Task05_Prostate")
    parser.add_argument("--output_root",  default="preprocessed/MSD")
    parser.add_argument("--no_vis",       action="store_true")
    args = parser.parse_args()
    metas = run_all(args.dataset_root, args.output_root, not args.no_vis)
    print(f"MSD preprocessing complete. Cases: {len(metas)}")

