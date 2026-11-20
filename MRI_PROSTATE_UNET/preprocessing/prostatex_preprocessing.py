"""
=============================================================================
prostatex_preprocessing.py
Dataset : PROSTATEx – PROSTATEx-Seg-HiRes
Format  : DICOM images + DICOM-SEG segmentation
MRI type: T2-weighted
Labels  : Whole-prostate gland segmentation (NOT lesion annotations)
=============================================================================

IMPORTANT:
  PROSTATEx includes two types of segmentation:
    (a) Lesion/finding annotations   – DO NOT USE
    (b) PROSTATEx-Seg-HiRes          – USE THIS (whole-gland, high-res)

  The Seg-HiRes collection provides DICOM-SEG objects with label 1 = whole
  prostate. Verify the segmentation type before processing.

Access: https://wiki.cancerimagingarchive.net/display/Public/PROSTATEx

Expected layout
---------------
datasets/PROSTATEx/
└── <PatientID>/
    ├── <StudyUID>/
    │   └── <SeriesUID>/           ← T2w DICOM .dcm files
    └── PROSTATEx-Seg-HiRes/
        └── <SeriesUID>/           ← DICOM-SEG .dcm files

Outputs
-------
preprocessed/PROSTATEx/<PatientID>/
    image_slice_<N>.npy
    mask_slice_<N>.npy
    meta.json
"""

import os
import json
import logging
import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import pydicom
import cv2
import matplotlib.pyplot as plt

try:
    import highdicom as hd
    HD_AVAILABLE = True
except ImportError:
    HD_AVAILABLE = False

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  [PROSTATEx]  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

TARGET_SIZE         = (256, 256)
CLIP_PERCENTILE     = (1, 99)
MIN_PROSTATE_PIXELS = 50
DATASET_TAG         = "PROSTATEx"

# Keywords to identify PROSTATEx-Seg-HiRes DICOM-SEG series
SEG_HIRES_KEYWORDS = ["seg-hires", "seg_hires", "seghires",
                       "prostatex-seg", "whole", "prostate seg"]


def _load_dicom_series(series_dir: Path):
    reader = sitk.ImageSeriesReader()
    names  = reader.GetGDCMSeriesFileNames(str(series_dir))
    if not names:
        raise ValueError(f"No DICOM found in {series_dir}")
    reader.SetFileNames(names)
    img = reader.Execute()
    arr = sitk.GetArrayFromImage(img).astype(np.float32)   # (D, H, W)
    return arr, img


def _is_seg_hires(dcm_path: str) -> bool:
    """Check if a DICOM file is a whole-gland PROSTATEx-Seg-HiRes object."""
    try:
        ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
        desc = str(getattr(ds, "SeriesDescription", "")).lower()
        modality = str(getattr(ds, "Modality", "")).upper()
        if modality == "SEG":
            if any(kw in desc for kw in SEG_HIRES_KEYWORDS):
                return True
            # Also check ContentDescription or label
            content = str(getattr(ds, "ContentDescription", "")).lower()
            if any(kw in content for kw in SEG_HIRES_KEYWORDS):
                return True
    except Exception:
        pass
    return False


def _find_t2w_and_seg(patient_dir: Path):
    """
    Walk patient folder and return:
        t2w_dir  : Path to T2w DICOM series directory
        seg_dcm  : Path to the DICOM-SEG file or directory
    """
    t2w_dir  = None
    seg_dcm  = None

    for root, dirs, files in os.walk(patient_dir):
        dcm_files = [os.path.join(root, f) for f in files
                     if f.lower().endswith(".dcm")]
        if not dcm_files:
            continue

        # Check if this is a SEG series
        if _is_seg_hires(dcm_files[0]):
            seg_dcm = Path(root)
            continue

        # Check modality = MR and T2w
        try:
            ds = pydicom.dcmread(dcm_files[0], stop_before_pixels=True)
            modality = str(getattr(ds, "Modality", "")).upper()
            desc     = str(getattr(ds, "SeriesDescription", "")).lower()

            if modality == "MR" and any(k in desc for k in
                                         ["t2", "t2w", "t2-weighted", "tra"]):
                if len(dcm_files) >= 5:   # must be a volume, not a single scan
                    t2w_dir = Path(root)
        except Exception:
            pass

    # Fallback: first MR series with ≥5 slices
    if t2w_dir is None:
        for root, dirs, files in os.walk(patient_dir):
            dcm_files = [f for f in files if f.lower().endswith(".dcm")]
            if len(dcm_files) >= 5:
                # exclude SEG directories
                try:
                    ds = pydicom.dcmread(os.path.join(root, dcm_files[0]),
                                         stop_before_pixels=True)
                    if str(getattr(ds, "Modality", "")).upper() == "MR":
                        t2w_dir = Path(root)
                        break
                except Exception:
                    pass

    return t2w_dir, seg_dcm


def _load_dicom_seg_mask(seg_dir: Path, ref_sitk_img) -> np.ndarray:
    """
    Load PROSTATEx-Seg-HiRes DICOM-SEG and resample to reference image grid.
    Returns binary mask (0/1).
    """
    if not HD_AVAILABLE:
        raise ImportError("highdicom not installed. Run: pip install highdicom")

    seg_files = sorted(seg_dir.glob("*.dcm"))
    if not seg_files:
        raise FileNotFoundError(f"No .dcm in {seg_dir}")

    seg = hd.seg.segread(str(seg_files[0]))
    mask_vol = np.zeros(ref_sitk_img.GetSize()[::-1], dtype=np.uint8)

    for seg_num in range(1, len(seg.SegmentSequence) + 1):
        frames = seg.get_pixels_by_segment(segment_numbers=[seg_num])
        seg_sitk = sitk.GetImageFromArray(frames[..., 0].astype(np.uint8))

        # Copy spatial metadata from reference
        seg_sitk.SetSpacing(ref_sitk_img.GetSpacing())
        seg_sitk.SetOrigin(ref_sitk_img.GetOrigin())
        seg_sitk.SetDirection(ref_sitk_img.GetDirection())

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(ref_sitk_img)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
        arr = sitk.GetArrayFromImage(resampler.Execute(seg_sitk))
        mask_vol[arr > 0] = 1   # binary – whole prostate only

    return mask_vol


def _normalize(volume: np.ndarray) -> np.ndarray:
    fg = volume[volume > 0]
    if fg.size == 0:
        return volume
    lo, hi = np.percentile(fg, CLIP_PERCENTILE)
    volume = np.clip(volume, lo, hi)
    if hi > lo:
        volume = (volume - lo) / (hi - lo)
    return volume.astype(np.float32)


def _resize_2d(arr, size, interp=cv2.INTER_LINEAR):
    return cv2.resize(arr.astype(np.float32), (size[1], size[0]),
                      interpolation=interp).astype(arr.dtype)


def preprocess_patient(patient_dir: Path,
                        out_dir: Path,
                        visualize: bool = True) -> dict:
    patient_id = patient_dir.name
    logger.info("Processing patient %s", patient_id)

    # ── 1. Locate T2w series and Seg-HiRes ───────────────────────────────
    t2w_dir, seg_dir = _find_t2w_and_seg(patient_dir)

    if t2w_dir is None:
        logger.warning("  ✗ T2w series not found for %s", patient_id)
        return {}
    if seg_dir is None:
        logger.warning("  ✗ Seg-HiRes not found for %s — "
                       "check for lesion-only annotations!", patient_id)
        return {}

    logger.info("  T2w : %s", t2w_dir)
    logger.info("  Seg : %s", seg_dir)

    # ── 2. Load T2w volume ────────────────────────────────────────────────
    try:
        image_vol, sitk_img = _load_dicom_series(t2w_dir)
    except Exception as e:
        logger.error("  ✗ DICOM load: %s", e)
        return {}

    # ── 3. Load whole-prostate segmentation ───────────────────────────────
    try:
        mask_vol = _load_dicom_seg_mask(seg_dir, sitk_img)
    except Exception as e:
        logger.error("  ✗ Seg-HiRes load: %s", e)
        return {}

    # ── 4. Shape check ────────────────────────────────────────────────────
    if image_vol.shape != mask_vol.shape:
        d = min(image_vol.shape[0], mask_vol.shape[0])
        if image_vol.shape[1:] == mask_vol.shape[1:]:
            image_vol = image_vol[:d]
            mask_vol  = mask_vol[:d]
        else:
            logger.error("  ✗ Shape mismatch — skipping %s", patient_id)
            return {}

    # Mask is already binary (whole gland = 1)
    mask_vol = (mask_vol > 0).astype(np.uint8)

    # ── 5. Normalise ──────────────────────────────────────────────────────
    image_vol = _normalize(image_vol)

    # ── 6. Select & save slices ───────────────────────────────────────────
    prostate_slices = [z for z in range(mask_vol.shape[0])
                       if mask_vol[z].sum() >= MIN_PROSTATE_PIXELS]
    context = set()
    for z in prostate_slices:
        for dz in [-1, 0, 1]:
            if 0 <= z + dz < mask_vol.shape[0]:
                context.add(z + dz)
    all_slices = sorted(context)

    if not all_slices:
        logger.warning("  ✗ No prostate slices for %s", patient_id)
        return {}

    out_dir.mkdir(parents=True, exist_ok=True)
    vis_data = []

    for z in all_slices:
        img_s = _resize_2d(image_vol[z], TARGET_SIZE, cv2.INTER_LINEAR)
        msk_s = _resize_2d(mask_vol[z].astype(np.float32),
                            TARGET_SIZE, cv2.INTER_NEAREST).astype(np.uint8)
        np.save(str(out_dir / f"image_slice_{z:04d}.npy"), img_s)
        np.save(str(out_dir / f"mask_slice_{z:04d}.npy"),  msk_s)
        if visualize and len(vis_data) < 4:
            vis_data.append((img_s, msk_s, z))

    meta = {
        "patient_id"    : patient_id,
        "dataset"       : DATASET_TAG,
        "t2w_dir"       : str(t2w_dir),
        "seg_dir"       : str(seg_dir),
        "original_shape": list(image_vol.shape),
        "n_slices_saved": len(all_slices),
        "spacing_mm"    : list(sitk_img.GetSpacing()),
        "target_size"   : list(TARGET_SIZE),
        "annotation"    : "PROSTATEx-Seg-HiRes whole-gland",
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    if visualize and vis_data:
        _save_verification_png(vis_data, out_dir, patient_id)

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
        axes[r, 0].imshow(img, cmap="gray");  axes[r, 0].set_title(f"T2w z={z}");    axes[r, 0].axis("off")
        axes[r, 1].imshow(msk, cmap="gray");  axes[r, 1].set_title("Whole-gland mask"); axes[r, 1].axis("off")
        axes[r, 2].imshow(ov);                axes[r, 2].set_title("Overlay");           axes[r, 2].axis("off")
    fig.suptitle(f"PROSTATEx – {title}")
    plt.tight_layout()
    plt.savefig(str(out_dir / "verification.png"), dpi=100)
    plt.close(fig)


def run_all(dataset_root: str, output_root: str,
            visualize: bool = True) -> list[dict]:
    root = Path(dataset_root)
    out  = Path(output_root)
    patient_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    logger.info("Found %d patient directories.", len(patient_dirs))

    all_meta = []
    for pdir in patient_dirs:
        meta = preprocess_patient(pdir, out / pdir.name, visualize=visualize)
        if meta:
            all_meta.append(meta)

    import pandas as pd
    pd.DataFrame(all_meta).to_csv(out / "prostatex_summary.csv", index=False)
    logger.info("PROSTATEx: processed %d / %d.", len(all_meta), len(patient_dirs))
    return all_meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="datasets/PROSTATEx")
    parser.add_argument("--output_root",  default="preprocessed/PROSTATEx")
    parser.add_argument("--no_vis",       action="store_true")
    args = parser.parse_args()
    metas = run_all(args.dataset_root, args.output_root, not args.no_vis)
    print(f"PROSTATEx preprocessing complete. Patients: {len(metas)}")

