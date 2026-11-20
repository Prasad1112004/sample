"""
=============================================================================
nci_3t_preprocessing.py
Dataset : NCI-ISBI 2013 Challenge – 3T Scanner subset
Format  : DICOM (.dcm) images + DICOM-SEG or NRRD (.nrrd) masks
MRI type: 3T T2-weighted
Labels  : CG = Central Gland (label 1), PZ = Peripheral Zone (label 2)
          → merged → binary whole-prostate mask
=============================================================================

Expected dataset folder layout (Tcia-style)
---------------------------------------------
datasets/NCI_ISBI_3T/
└── <PatientID>/
    ├── <StudyUID>/
    │   └── <SeriesUID>/        ← DICOM .dcm files
    └── Segmentation/
        └── <SegSeries>/        ← DICOM-SEG .dcm  OR  segmentation.nrrd

Access: https://wiki.cancerimagingarchive.net/display/Public/NCI-ISBI+2013+Challenge

Outputs
-------
preprocessed/NCI_ISBI_3T/<PatientID>/
    image_slice_<N>.npy   (float32, 256×256, normalised)
    mask_slice_<N>.npy    (uint8,   256×256, binary 0/1)
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
    import nrrd          # pynrrd
    NRRD_AVAILABLE = True
except ImportError:
    NRRD_AVAILABLE = False

try:
    import highdicom as hd   # DICOM-SEG reader
    HD_AVAILABLE = True
except ImportError:
    HD_AVAILABLE = False

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  [NCI_3T]  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

TARGET_SIZE        = (256, 256)
CLIP_PERCENTILE    = (1, 99)
MIN_PROSTATE_PIXELS = 50
DATASET_TAG        = "NCI_ISBI_3T"


# ─────────────────────────────────────────────────────────────────────────────
# DICOM image loading via SimpleITK
# ─────────────────────────────────────────────────────────────────────────────
def _load_dicom_series(series_dir: Path) -> tuple[np.ndarray, dict, sitk.Image]:
    """
    Read a DICOM series directory with SimpleITK.
    Returns (array [D,H,W] float32, metadata dict, SimpleITK image object).
    """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(series_dir))
    if not dicom_names:
        raise ValueError(f"No DICOM series found in {series_dir}")
    reader.SetFileNames(dicom_names)
    img = reader.Execute()
    arr = sitk.GetArrayFromImage(img).astype(np.float32)   # (D, H, W)
    meta = {
        "spacing"  : list(img.GetSpacing()),
        "origin"   : list(img.GetOrigin()),
        "direction": list(img.GetDirection()),
    }
    return arr, meta, img


# ─────────────────────────────────────────────────────────────────────────────
# Mask loading – NRRD or DICOM-SEG
# ─────────────────────────────────────────────────────────────────────────────
def _load_nrrd_mask(nrrd_path: Path, ref_sitk_img) -> np.ndarray:
    """Load NRRD segmentation and resample to reference image grid."""
    if not NRRD_AVAILABLE:
        raise ImportError("pynrrd not installed. Run: pip install pynrrd")
    data, header = nrrd.read(str(nrrd_path))   # shape varies

    # Convert to SimpleITK image for resampling
    seg_sitk = sitk.GetImageFromArray(data.astype(np.uint8))

    # Copy spacing/origin/direction from header if present
    if "spacings" in header:
        seg_sitk.SetSpacing(header["spacings"][::-1].tolist())
    if "space origin" in header:
        seg_sitk.SetOrigin(header["space origin"].tolist())

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_sitk_img)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampled = resampler.Execute(seg_sitk)
    return sitk.GetArrayFromImage(resampled).astype(np.uint8)


def _load_dicom_seg(seg_dir: Path, ref_sitk_img) -> np.ndarray:
    """Load DICOM-SEG file(s) using highdicom, resample to reference grid."""
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
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(ref_sitk_img)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampled = resampler.Execute(seg_sitk)
        arr = sitk.GetArrayFromImage(resampled)
        mask_vol[arr > 0] = seg_num

    return mask_vol


def _find_t2w_series(patient_dir: Path) -> Path | None:
    """
    Heuristic: find the DICOM series sub-folder most likely to be T2w.
    Checks SeriesDescription DICOM tag.
    """
    for root, dirs, files in os.walk(patient_dir):
        dcm_files = [f for f in files if f.lower().endswith(".dcm")]
        if dcm_files:
            try:
                ds = pydicom.dcmread(os.path.join(root, dcm_files[0]),
                                     stop_before_pixels=True)
                desc = str(getattr(ds, "SeriesDescription", "")).lower()
                if any(kw in desc for kw in ["t2", "t2w", "t2-weighted",
                                              "tra", "transversal"]):
                    return Path(root)
            except Exception:
                pass
    # Fallback: return first folder with ≥5 DICOM files
    for root, dirs, files in os.walk(patient_dir):
        dcm_files = [f for f in files if f.lower().endswith(".dcm")]
        if len(dcm_files) >= 5:
            return Path(root)
    return None


def _find_seg_resource(patient_dir: Path) -> tuple[Path | None, str]:
    """Return (path, type) where type is 'nrrd' or 'dicom_seg'."""
    nrrd_files = list(patient_dir.rglob("*.nrrd"))
    if nrrd_files:
        return nrrd_files[0], "nrrd"
    # DICOM-SEG: look for folder named 'Segmentation' or similar
    for root, dirs, _ in os.walk(patient_dir):
        if "seg" in root.lower() or "label" in root.lower():
            dcm_files = list(Path(root).glob("*.dcm"))
            if dcm_files:
                return Path(root), "dicom_seg"
    return None, "unknown"


def _normalize(volume: np.ndarray) -> np.ndarray:
    fg = volume[volume > 0]
    if fg.size == 0:
        return volume
    lo, hi = np.percentile(fg, CLIP_PERCENTILE)
    volume = np.clip(volume, lo, hi)
    if hi > lo:
        volume = (volume - lo) / (hi - lo)
    return volume.astype(np.float32)


def _binarize(mask: np.ndarray) -> np.ndarray:
    """CG(1) + PZ(2) → whole-prostate(1)."""
    return ((mask == 1) | (mask == 2)).astype(np.uint8)


def _resize_2d(arr, size, interp=cv2.INTER_LINEAR):
    return cv2.resize(arr.astype(np.float32), (size[1], size[0]),
                      interpolation=interp).astype(arr.dtype)


def preprocess_patient(patient_dir: Path,
                        out_dir: Path,
                        visualize: bool = True) -> dict:
    patient_id = patient_dir.name
    logger.info("Processing patient %s", patient_id)

    # ── 1. Find T2w DICOM series ───────────────────────────────────────────
    t2w_dir = _find_t2w_series(patient_dir)
    if t2w_dir is None:
        logger.warning("  ✗ T2w series not found for %s", patient_id)
        return {}

    logger.info("  T2w dir: %s", t2w_dir)

    # ── 2. Load DICOM image volume ─────────────────────────────────────────
    try:
        image_vol, meta_dict, sitk_img = _load_dicom_series(t2w_dir)
    except Exception as e:
        logger.error("  ✗ DICOM load error: %s", e)
        return {}

    # ── 3. Load segmentation ───────────────────────────────────────────────
    seg_path, seg_type = _find_seg_resource(patient_dir)
    if seg_path is None:
        logger.warning("  ✗ No segmentation found for %s", patient_id)
        return {}

    try:
        if seg_type == "nrrd":
            mask_vol = _load_nrrd_mask(seg_path, sitk_img)
        elif seg_type == "dicom_seg":
            mask_vol = _load_dicom_seg(seg_path, sitk_img)
        else:
            logger.warning("  ✗ Unknown seg format for %s", patient_id)
            return {}
    except Exception as e:
        logger.error("  ✗ Seg load error: %s", e)
        return {}

    # ── 4. Shape alignment ────────────────────────────────────────────────
    if image_vol.shape != mask_vol.shape:
        logger.warning("  Shape mismatch %s vs %s — attempting fix.",
                       image_vol.shape, mask_vol.shape)
        if image_vol.shape[1:] == mask_vol.shape[1:]:
            # Slice count differs — crop/pad along axis 0
            d = min(image_vol.shape[0], mask_vol.shape[0])
            image_vol = image_vol[:d]
            mask_vol  = mask_vol[:d]
        else:
            logger.error("  ✗ Irrecoverable shape mismatch — skipping.")
            return {}

    # ── 5. Binarize ───────────────────────────────────────────────────────
    mask_vol = _binarize(mask_vol)

    # ── 6. Normalise ──────────────────────────────────────────────────────
    image_vol = _normalize(image_vol)

    # ── 7. Select & save slices ───────────────────────────────────────────
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

    # ── 8. Metadata ───────────────────────────────────────────────────────
    meta = {
        "patient_id"    : patient_id,
        "dataset"       : DATASET_TAG,
        "t2w_dir"       : str(t2w_dir),
        "seg_path"      : str(seg_path),
        "seg_type"      : seg_type,
        "original_shape": list(image_vol.shape),
        "n_slices_saved": len(all_slices),
        "spacing_mm"    : meta_dict.get("spacing", []),
        "target_size"   : list(TARGET_SIZE),
        "labels_merged" : "CG(1) + PZ(2) → 1",
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
        axes[r, 0].imshow(img, cmap="gray");  axes[r, 0].set_title(f"T2w (3T) z={z}"); axes[r, 0].axis("off")
        axes[r, 1].imshow(msk, cmap="gray");  axes[r, 1].set_title("Binary mask");      axes[r, 1].axis("off")
        axes[r, 2].imshow(ov);                axes[r, 2].set_title("Overlay");           axes[r, 2].axis("off")
    fig.suptitle(f"NCI-ISBI 3T – {title}")
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
    pd.DataFrame(all_meta).to_csv(out / "nci_3t_summary.csv", index=False)
    logger.info("NCI_3T: processed %d / %d.", len(all_meta), len(patient_dirs))
    return all_meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="datasets/NCI_ISBI_3T")
    parser.add_argument("--output_root",  default="preprocessed/NCI_ISBI_3T")
    parser.add_argument("--no_vis",       action="store_true")
    args = parser.parse_args()
    metas = run_all(args.dataset_root, args.output_root, not args.no_vis)
    print(f"NCI 3T preprocessing complete. Patients: {len(metas)}")

