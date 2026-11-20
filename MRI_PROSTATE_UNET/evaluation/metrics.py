"""
=============================================================================
metrics.py  –  Evaluation Metrics for Prostate MRI Segmentation
=============================================================================

Implements all eight evaluation parameters:
  1. Dice Similarity Coefficient (DSC)
  2. Intersection over Union (IoU)
  3. Accuracy
  4. Precision
  5. Recall (= Sensitivity)
  6. F1 Score
  7. Sensitivity  (same as Recall for binary segmentation)
  8. Specificity

All functions operate on flattened binary arrays (0/1) or batch tensors.
"""

import numpy as np
import torch
from typing import Dict, Tuple


EPS = 1e-7   # prevent division by zero


# ─────────────────────────────────────────────────────────────────────────────
# NumPy-level metrics (work on flat 1D arrays, 0/1 uint8)
# ─────────────────────────────────────────────────────────────────────────────
def compute_confusion(pred: np.ndarray,
                       gt:   np.ndarray) -> Tuple[int, int, int, int]:
    """
    Compute TP, TN, FP, FN from binary predictions and ground truth.

    Parameters
    ----------
    pred : (N,)  Binary prediction (0 or 1)
    gt   : (N,)  Binary ground truth (0 or 1)

    Returns
    -------
    TP, TN, FP, FN
    """
    pred = pred.astype(bool)
    gt   = gt.astype(bool)

    TP = int(np.logical_and(pred,  gt ).sum())
    TN = int(np.logical_and(~pred, ~gt).sum())
    FP = int(np.logical_and(pred,  ~gt).sum())
    FN = int(np.logical_and(~pred,  gt).sum())
    return TP, TN, FP, FN


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice Similarity Coefficient = 2*TP / (2*TP + FP + FN)."""
    TP, TN, FP, FN = compute_confusion(pred.flatten(), gt.flatten())
    return (2 * TP) / (2 * TP + FP + FN + EPS)


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Intersection over Union = TP / (TP + FP + FN)."""
    TP, TN, FP, FN = compute_confusion(pred.flatten(), gt.flatten())
    return TP / (TP + FP + FN + EPS)


def accuracy(pred: np.ndarray, gt: np.ndarray) -> float:
    """Pixel Accuracy = (TP + TN) / (TP + TN + FP + FN)."""
    TP, TN, FP, FN = compute_confusion(pred.flatten(), gt.flatten())
    return (TP + TN) / (TP + TN + FP + FN + EPS)


def precision(pred: np.ndarray, gt: np.ndarray) -> float:
    """Precision = TP / (TP + FP)."""
    TP, TN, FP, FN = compute_confusion(pred.flatten(), gt.flatten())
    return TP / (TP + FP + EPS)


def recall(pred: np.ndarray, gt: np.ndarray) -> float:
    """Recall (= Sensitivity) = TP / (TP + FN)."""
    TP, TN, FP, FN = compute_confusion(pred.flatten(), gt.flatten())
    return TP / (TP + FN + EPS)


def f1_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    F1 Score = harmonic mean of Precision and Recall.
    For binary segmentation, F1 == Dice.
    """
    p = precision(pred, gt)
    r = recall(pred, gt)
    return 2 * p * r / (p + r + EPS)


def sensitivity(pred: np.ndarray, gt: np.ndarray) -> float:
    """Sensitivity = TP / (TP + FN)  (identical to Recall)."""
    return recall(pred, gt)


def specificity(pred: np.ndarray, gt: np.ndarray) -> float:
    """Specificity = TN / (TN + FP)."""
    TP, TN, FP, FN = compute_confusion(pred.flatten(), gt.flatten())
    return TN / (TN + FP + EPS)


def compute_all_metrics(pred: np.ndarray,
                         gt:   np.ndarray) -> Dict[str, float]:
    """
    Compute all eight metrics for one prediction-GT pair.

    Parameters
    ----------
    pred : 2D or 3D binary array (H, W) or (D, H, W)
    gt   : same shape as pred

    Returns
    -------
    dict  {metric_name: value}
    """
    pred_flat = pred.flatten()
    gt_flat   = gt.flatten()

    return {
        "Dice"       : dice_score(pred_flat, gt_flat),
        "IoU"        : iou_score(pred_flat, gt_flat),
        "Accuracy"   : accuracy(pred_flat, gt_flat),
        "Precision"  : precision(pred_flat, gt_flat),
        "Recall"     : recall(pred_flat, gt_flat),
        "F1"         : f1_score(pred_flat, gt_flat),
        "Sensitivity": sensitivity(pred_flat, gt_flat),
        "Specificity": specificity(pred_flat, gt_flat),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Batch-level PyTorch metrics (used during training validation)
# ─────────────────────────────────────────────────────────────────────────────
def dice_score_batch(preds: torch.Tensor,
                      targets: torch.Tensor,
                      threshold: float = 0.5,
                      smooth: float = 1.0) -> float:
    """
    Compute mean Dice score for a batch of predictions (logits or probs).

    Parameters
    ----------
    preds   : (B, 1, H, W) – raw logits
    targets : (B, 1, H, W) – binary ground-truth
    """
    probs = torch.sigmoid(preds)
    preds_bin = (probs > threshold).float()

    preds_flat   = preds_bin.view(preds_bin.shape[0], -1)
    targets_flat = targets.view(targets.shape[0], -1)

    intersection = (preds_flat * targets_flat).sum(dim=1)
    dice_per_img = (2 * intersection + smooth) / \
                   (preds_flat.sum(dim=1) + targets_flat.sum(dim=1) + smooth)
    return dice_per_img.mean().item()


def iou_score_batch(preds: torch.Tensor,
                     targets: torch.Tensor,
                     threshold: float = 0.5,
                     smooth: float = 1.0) -> float:
    """Compute mean IoU for a batch."""
    probs = torch.sigmoid(preds)
    preds_bin = (probs > threshold).float()

    preds_flat   = preds_bin.view(preds_bin.shape[0], -1)
    targets_flat = targets.view(targets.shape[0], -1)

    intersection = (preds_flat * targets_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1) - intersection
    iou_per_img = (intersection + smooth) / (union + smooth)
    return iou_per_img.mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# Summary statistics helper
# ─────────────────────────────────────────────────────────────────────────────
def aggregate_metrics(metric_list: list[dict]) -> dict:
    """
    Given a list of per-patient metric dicts, compute mean ± std for each
    metric.

    Returns
    -------
    dict  {metric_name: {"mean": float, "std": float, "values": list}}
    """
    if not metric_list:
        return {}

    keys = list(metric_list[0].keys())
    summary = {}
    for k in keys:
        vals = [m[k] for m in metric_list if k in m]
        summary[k] = {
            "mean"  : float(np.mean(vals)),
            "std"   : float(np.std(vals)),
            "values": vals,
        }
    return summary


def print_metrics_table(summary: dict, dataset_name: str = "Overall"):
    """Print a formatted metrics table to stdout."""
    print(f"\n{'═'*55}")
    print(f"  Evaluation Results – {dataset_name}")
    print(f"{'═'*55}")
    print(f"  {'Metric':<16} {'Mean':>8}  {'Std':>8}")
    print(f"  {'-'*40}")
    for metric, vals in summary.items():
        print(f"  {metric:<16} {vals['mean']:>8.4f}  {vals['std']:>8.4f}")
    print(f"{'═'*55}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    pred = (np.random.rand(256, 256) > 0.4).astype(np.uint8)
    gt   = (np.random.rand(256, 256) > 0.5).astype(np.uint8)

    metrics = compute_all_metrics(pred, gt)
    print("Single-image metrics:")
    for k, v in metrics.items():
        print(f"  {k:<12} : {v:.4f}")

    # Batch test
    preds_t = torch.randn(4, 1, 256, 256)
    tgts_t  = torch.randint(0, 2, (4, 1, 256, 256)).float()
    print(f"\nBatch Dice : {dice_score_batch(preds_t, tgts_t):.4f}")
    print(f"Batch IoU  : {iou_score_batch(preds_t,  tgts_t):.4f}")

