"""
=============================================================================
evaluate.py  –  Full Model Evaluation on the Test Set
=============================================================================

Usage:
  python evaluation/evaluate.py
         --preprocessed_root preprocessed/
         --checkpoint         results/models/combined_prostate_unet.pth
         --test_csv           results/split_test.csv
         --output_dir         results/

Generates:
  • results/metrics.csv              – per-patient metrics
  • results/graphs/                  – all performance visualisation graphs
  • results/predictions/             – segmentation overlay images
  • Console print of five-dataset comparison table
"""

import sys
import json
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.unet import build_unet
from evaluation.metrics import (compute_all_metrics, aggregate_metrics,
                                 print_metrics_table)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [EVAL]  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

DATASET_NAMES = ["I2CVB", "MSD", "NCI_ISBI_3T", "NCI_ISBI_Dx", "PROSTATEx"]
METRICS_ORDER = ["Dice", "IoU", "Accuracy", "Precision",
                 "Recall", "F1", "Sensitivity", "Specificity"]


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_slice(model, img_path: str, device, threshold: float = 0.5):
    """
    Load one preprocessed slice and return (image_np, pred_mask_np).
    """
    img_np = np.load(img_path).astype(np.float32)          # (H, W)
    img_t  = torch.from_numpy(img_np[np.newaxis, np.newaxis]).to(device)  # (1,1,H,W)
    logits = model(img_t)
    probs  = torch.sigmoid(logits)
    pred   = (probs[0, 0].cpu().numpy() > threshold).astype(np.uint8)
    return img_np, pred


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Loop
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── 1. Load model ─────────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = build_unet(device=str(device))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info("Loaded checkpoint from epoch %d (val_loss=%.4f)",
                ckpt.get("epoch", 0), ckpt.get("val_loss", float("nan")))

    # ── 2. Load test set ──────────────────────────────────────────────────
    test_df = pd.read_csv(args.test_csv)
    logger.info("Test slices: %d", len(test_df))

    # ── 3. Per-patient aggregation ─────────────────────────────────────────
    # Group slices by (dataset, patient_id) and compute 3D-level metrics
    patient_records = []
    pred_save_dir   = Path(args.output_dir) / "predictions"
    pred_save_dir.mkdir(parents=True, exist_ok=True)

    groups = test_df.groupby(["dataset", "patient_id"])
    n_groups = len(groups)
    logger.info("Evaluating %d patients …", n_groups)

    for g_idx, ((ds, pid), group) in enumerate(groups):
        slices_sorted = group.sort_values("image_path")
        all_preds, all_gts = [], []
        sample_img, sample_pred, sample_gt = None, None, None

        for _, row in slices_sorted.iterrows():
            img_np, pred_np = predict_slice(model, row["image_path"], device)
            gt_np = np.load(row["mask_path"]).astype(np.uint8)

            all_preds.append(pred_np)
            all_gts.append(gt_np)

            if sample_img is None:   # save first slice for visualisation
                sample_img  = img_np
                sample_pred = pred_np
                sample_gt   = gt_np

        # Stack to 3D volume for volume-level metrics
        pred_vol = np.stack(all_preds, axis=0)
        gt_vol   = np.stack(all_gts,   axis=0)
        metrics  = compute_all_metrics(pred_vol, gt_vol)
        metrics.update({"dataset": ds, "patient_id": pid})
        patient_records.append(metrics)

        # ── Save sample visualisation ──────────────────────────────────────
        _save_prediction_png(sample_img, sample_gt, sample_pred,
                              pred_save_dir / f"{ds}_{pid}.png",
                              title=f"{ds} – {pid}")

        if (g_idx + 1) % 10 == 0:
            logger.info("  … %d / %d patients done.", g_idx + 1, n_groups)

    # ── 4. Build results DataFrame ─────────────────────────────────────────
    results_df = pd.DataFrame(patient_records)
    results_df.to_csv(Path(args.output_dir) / "metrics.csv", index=False)
    logger.info("Per-patient metrics → %s",
                Path(args.output_dir) / "metrics.csv")

    # ── 5. Print & save summary tables ────────────────────────────────────
    graphs_dir = Path(args.output_dir) / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    # Overall summary
    overall_metrics = [
        {k: row[k] for k in METRICS_ORDER}
        for _, row in results_df.iterrows()
    ]
    overall_summary = aggregate_metrics(overall_metrics)
    print_metrics_table(overall_summary, "Overall")

    # Per-dataset summary
    dataset_summaries = {}
    for ds in results_df["dataset"].unique():
        ds_rows = results_df[results_df["dataset"] == ds]
        ds_metrics = [{k: row[k] for k in METRICS_ORDER}
                      for _, row in ds_rows.iterrows()]
        ds_summary = aggregate_metrics(ds_metrics)
        dataset_summaries[ds] = ds_summary
        print_metrics_table(ds_summary, ds)

    # ── 6. Generate all graphs ─────────────────────────────────────────────
    _plot_dataset_comparison(dataset_summaries, overall_summary, graphs_dir)
    _plot_eight_metrics_bar(overall_summary, graphs_dir)
    _plot_precision_recall(dataset_summaries, overall_summary, graphs_dir)

    # Training curves (if history available)
    hist_path = Path(args.output_dir) / "training_history.json"
    if hist_path.exists():
        with open(hist_path) as f:
            history = json.load(f)
        _plot_training_curves_from_history(history, graphs_dir)

    # ── 7. Five-dataset comparison table ──────────────────────────────────
    _print_comparison_table(dataset_summaries, overall_summary)

    logger.info("\n✅ Evaluation complete.  Graphs → %s", graphs_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────
def _save_prediction_png(img, gt, pred, out_path, title=""):
    """Save a 4-panel: original MRI | GT mask | predicted mask | overlay."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    overlay = np.stack([img, img, img], axis=-1)
    # GT in green, prediction in red
    overlay[gt   == 1, 1] = 1.0
    overlay[pred == 1, 0] = 1.0

    axes[0].imshow(img,     cmap="gray");  axes[0].set_title("MRI (T2w)");          axes[0].axis("off")
    axes[1].imshow(gt,      cmap="gray");  axes[1].set_title("Ground Truth Mask");   axes[1].axis("off")
    axes[2].imshow(pred,    cmap="gray");  axes[2].set_title("Predicted Mask");       axes[2].axis("off")
    axes[3].imshow(overlay);               axes[3].set_title("Overlay (GT=G,Pred=R)"); axes[3].axis("off")

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120)
    plt.close(fig)


def _plot_dataset_comparison(dataset_summaries: dict,
                               overall_summary: dict,
                               graphs_dir: Path):
    """Bar-chart: Dice, IoU, Accuracy for each dataset + overall."""
    datasets   = list(dataset_summaries.keys()) + ["Overall"]
    dice_means = [dataset_summaries[d]["Dice"]["mean"] for d in list(dataset_summaries)] \
                 + [overall_summary["Dice"]["mean"]]
    iou_means  = [dataset_summaries[d]["IoU"]["mean"]  for d in list(dataset_summaries)] \
                 + [overall_summary["IoU"]["mean"]]
    acc_means  = [dataset_summaries[d]["Accuracy"]["mean"] for d in list(dataset_summaries)] \
                 + [overall_summary["Accuracy"]["mean"]]

    x    = np.arange(len(datasets))
    w    = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w,  dice_means, w, label="Dice",     color="steelblue")
    ax.bar(x,      iou_means,  w, label="IoU",      color="coral")
    ax.bar(x + w,  acc_means,  w, label="Accuracy", color="seagreen")
    ax.set_xticks(x);  ax.set_xticklabels(datasets, rotation=15)
    ax.set_ylabel("Score");  ax.set_title("Dataset-wise Dice, IoU & Accuracy Comparison")
    ax.set_ylim(0, 1.05);   ax.legend();  ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(graphs_dir / "dataset_comparison.png"), dpi=150)
    plt.close(fig)
    logger.info("Saved dataset_comparison.png")


def _plot_eight_metrics_bar(overall_summary: dict, graphs_dir: Path):
    """Horizontal bar chart of all 8 metrics (mean ± std) for overall model."""
    metrics = METRICS_ORDER
    means   = [overall_summary[m]["mean"] for m in metrics]
    stds    = [overall_summary[m]["std"]  for m in metrics]

    colours = ["#4e79a7","#f28e2b","#e15759","#76b7b2",
               "#59a14f","#edc948","#b07aa1","#ff9da7"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(metrics, means, xerr=stds, color=colours,
                   capsize=5, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, means):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Score")
    ax.set_title("Combined U-Net – All 8 Evaluation Metrics (Overall)")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(graphs_dir / "eight_metrics_overall.png"), dpi=150)
    plt.close(fig)
    logger.info("Saved eight_metrics_overall.png")


def _plot_precision_recall(dataset_summaries: dict,
                            overall_summary: dict,
                            graphs_dir: Path):
    """Precision & Recall grouped bar chart per dataset."""
    all_ds     = list(dataset_summaries.keys()) + ["Overall"]
    all_sum    = {**dataset_summaries, "Overall": overall_summary}
    prec_means = [all_sum[d]["Precision"]["mean"] for d in all_ds]
    rec_means  = [all_sum[d]["Recall"]["mean"]    for d in all_ds]

    x = np.arange(len(all_ds))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, prec_means, w, label="Precision", color="royalblue")
    ax.bar(x + w/2, rec_means,  w, label="Recall",    color="tomato")
    ax.set_xticks(x);  ax.set_xticklabels(all_ds, rotation=15)
    ax.set_ylim(0, 1.05);  ax.legend();  ax.grid(axis="y", alpha=0.3)
    ax.set_title("Precision & Recall Comparison")
    plt.tight_layout()
    plt.savefig(str(graphs_dir / "precision_recall_comparison.png"), dpi=150)
    plt.close(fig)
    logger.info("Saved precision_recall_comparison.png")


def _plot_training_curves_from_history(history: dict, graphs_dir: Path):
    """Reproduce training + validation loss / dice graphs from saved history."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="steelblue")
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   color="coral")
    axes[0].set_title("Training & Validation Loss")
    axes[0].set_xlabel("Epoch");  axes[0].set_ylabel("Loss")
    axes[0].legend();  axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_dice"], label="Train Dice", color="steelblue")
    axes[1].plot(epochs, history["val_dice"],   label="Val Dice",   color="coral")
    axes[1].set_title("Dice Score over Epochs")
    axes[1].set_xlabel("Epoch");  axes[1].set_ylabel("Dice")
    axes[1].legend();  axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(graphs_dir / "loss_dice_curves.png"), dpi=150)
    plt.close(fig)
    logger.info("Saved loss_dice_curves.png")


def _print_comparison_table(dataset_summaries: dict, overall_summary: dict):
    """Print a formatted five-dataset comparison table."""
    print("\n" + "═" * 90)
    print("  FIVE-DATASET COMPARISON TABLE   (mean ± std)")
    print("═" * 90)
    header = f"  {'Dataset':<18}" + "".join(f" {'M':>10}" for M in METRICS_ORDER)
    header = header.replace("M", "{:>10}".format)
    # Proper header
    col_str = "  {:<18}".format("Dataset")
    for m in METRICS_ORDER:
        col_str += "  {:>10}".format(m[:10])
    print(col_str)
    print("-" * 90)

    all_sources = list(dataset_summaries.keys()) + ["Overall"]
    all_data    = {**dataset_summaries, "Overall": overall_summary}

    for src in all_sources:
        summary = all_data[src]
        row_str = "  {:<18}".format(src)
        for m in METRICS_ORDER:
            val = summary[m]["mean"]
            std = summary[m]["std"]
            row_str += "  {:>10}".format(f"{val:.3f}±{std:.3f}"[:10])
        print(row_str)

    print("═" * 90 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained U-Net on the prostate MRI test set."
    )
    parser.add_argument("--preprocessed_root", type=str, default="preprocessed")
    parser.add_argument("--checkpoint",        type=str,
                        default="results/models/combined_prostate_unet.pth")
    parser.add_argument("--test_csv",          type=str,
                        default="results/split_test.csv")
    parser.add_argument("--output_dir",        type=str, default="results")
    parser.add_argument("--threshold",         type=float, default=0.5)
    args = parser.parse_args()
    evaluate(args)

