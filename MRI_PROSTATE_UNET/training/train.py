"""
=============================================================================
train.py  –  Combined U-Net Training Pipeline for Prostate MRI Segmentation
=============================================================================

Usage (from project root):
  python training/train.py --preprocessed_root preprocessed/
                           --output_dir       results/
                           --epochs 30 --batch_size 8 --lr 0.001

Google Colab:
  !python training/train.py --preprocessed_root /content/drive/MyDrive/preprocessed/
                             --output_dir       /content/drive/MyDrive/results/
"""

import os
import sys
import json
import logging
import argparse
import random
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.unet import build_unet, CombinedLoss
from evaluation.metrics import dice_score_batch, iou_score_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [TRAIN]  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

DATASET_NAMES = ["I2CVB", "MSD", "NCI_ISBI_3T", "NCI_ISBI_Dx", "PROSTATEx"]


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Discovery & Patient-Level Splitting
# ─────────────────────────────────────────────────────────────────────────────
def discover_slices(preprocessed_root: Path) -> pd.DataFrame:
    """
    Walk all preprocessed dataset folders and collect slice paths.

    Returns DataFrame columns:
        dataset, patient_id, image_path, mask_path
    """
    records = []
    for ds in DATASET_NAMES:
        ds_dir = preprocessed_root / ds
        if not ds_dir.exists():
            logger.warning("Preprocessed folder not found: %s — skipping.", ds_dir)
            continue
        for patient_dir in sorted(ds_dir.iterdir()):
            if not patient_dir.is_dir():
                continue
            img_files  = sorted(patient_dir.glob("image_slice_*.npy"))
            mask_files = sorted(patient_dir.glob("mask_slice_*.npy"))
            if not img_files:
                continue
            for img_f in img_files:
                # Match corresponding mask
                mask_f = patient_dir / img_f.name.replace("image_", "mask_")
                if not mask_f.exists():
                    logger.debug("  No mask for %s — skipping slice.", img_f)
                    continue
                records.append({
                    "dataset"   : ds,
                    "patient_id": patient_dir.name,
                    "image_path": str(img_f),
                    "mask_path" : str(mask_f),
                })
    df = pd.DataFrame(records)
    logger.info("Total slices discovered: %d across %d datasets.",
                len(df), df["dataset"].nunique())
    return df


def patient_split(df: pd.DataFrame,
                  train_frac: float = 0.70,
                  val_frac:   float = 0.15,
                  seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Patient-level train / val / test split.
    No patient appears in more than one subset.
    NCI_ISBI_3T and NCI_ISBI_Dx share patient IDs are handled by
    dataset-specific splitting.
    """
    rng = random.Random(seed)
    train_records, val_records, test_records = [], [], []

    for ds in df["dataset"].unique():
        ds_df     = df[df["dataset"] == ds]
        patients  = sorted(ds_df["patient_id"].unique())
        rng.shuffle(patients)

        n        = len(patients)
        n_train  = max(1, int(n * train_frac))
        n_val    = max(1, int(n * val_frac))

        train_pts = patients[:n_train]
        val_pts   = patients[n_train: n_train + n_val]
        test_pts  = patients[n_train + n_val:]

        train_records.append(ds_df[ds_df["patient_id"].isin(train_pts)])
        val_records.append(  ds_df[ds_df["patient_id"].isin(val_pts)])
        test_records.append( ds_df[ds_df["patient_id"].isin(test_pts)])

        logger.info("  %-15s patients: total=%d  train=%d  val=%d  test=%d",
                    ds, n, len(train_pts), len(val_pts), len(test_pts))

    train_df = pd.concat(train_records, ignore_index=True)
    val_df   = pd.concat(val_records,   ignore_index=True)
    test_df  = pd.concat(test_records,  ignore_index=True)

    # Verify no patient leakage
    for ds in df["dataset"].unique():
        tr = set(train_df[train_df["dataset"]==ds]["patient_id"])
        vl = set(val_df  [val_df["dataset"]==ds]["patient_id"])
        te = set(test_df [test_df["dataset"]==ds]["patient_id"])
        assert not (tr & vl), f"Train/Val patient overlap in {ds}"
        assert not (tr & te), f"Train/Test patient overlap in {ds}"
        assert not (vl & te), f"Val/Test patient overlap in {ds}"

    logger.info("\nSplit summary  train=%d  val=%d  test=%d  total=%d",
                len(train_df), len(val_df), len(test_df), len(df))
    return train_df, val_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────
class ProstateSliceDataset(Dataset):
    """
    Loads preprocessed 256×256 prostate MRI slices (NumPy arrays).

    Each item: (image_tensor [1,H,W] float32, mask_tensor [1,H,W] float32)
    """

    def __init__(self, df: pd.DataFrame, augment: bool = False):
        self.records = df.reset_index(drop=True)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        row = self.records.iloc[idx]
        img  = np.load(row["image_path"]).astype(np.float32)   # (H, W)
        mask = np.load(row["mask_path"]).astype(np.float32)    # (H, W)

        # Basic sanity check
        assert img.shape  == (256, 256), f"Unexpected img shape {img.shape}"
        assert mask.shape == (256, 256), f"Unexpected mask shape {mask.shape}"

        # Add channel dimension
        img  = img[np.newaxis, ...]    # (1, H, W)
        mask = mask[np.newaxis, ...]   # (1, H, W)

        # ── Data augmentation (training only) ──────────────────────────────
        if self.augment:
            img, mask = self._augment(img, mask)

        return (
            torch.from_numpy(img),
            torch.from_numpy(mask),
            row["dataset"],       # preserve dataset label for evaluation
            row["patient_id"],
        )

    @staticmethod
    def _augment(img: np.ndarray, mask: np.ndarray):
        """Random horizontal / vertical flip and 90° rotation."""
        if random.random() > 0.5:
            img  = np.flip(img,  axis=2).copy()
            mask = np.flip(mask, axis=2).copy()
        if random.random() > 0.5:
            img  = np.flip(img,  axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        k = random.randint(0, 3)
        img  = np.rot90(img,  k=k, axes=(1, 2)).copy()
        mask = np.rot90(mask, k=k, axes=(1, 2)).copy()
        return img, mask


# ─────────────────────────────────────────────────────────────────────────────
# Dataset-Aware Weighted Sampler
# ─────────────────────────────────────────────────────────────────────────────
def build_weighted_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    """
    Assign each slice a weight inversely proportional to its dataset size,
    so smaller datasets are not dominated by larger ones.
    """
    dataset_counts = df["dataset"].value_counts().to_dict()
    n_datasets = len(dataset_counts)
    weights = []
    for _, row in df.iterrows():
        count = dataset_counts[row["dataset"]]
        # Weight = (1 / count) * n_datasets  → equalises across datasets
        weights.append(n_datasets / count)
    sampler = WeightedRandomSampler(weights=weights,
                                     num_samples=len(weights),
                                     replacement=True)
    return sampler


# ─────────────────────────────────────────────────────────────────────────────
# Training / Validation Loops
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler=None):
    model.train()
    total_loss = 0.0
    total_dice = 0.0

    for imgs, masks, _, _ in loader:
        imgs  = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        if scaler is not None:   # AMP
            with torch.cuda.amp.autocast():
                logits = model(imgs)
                loss   = loss_fn(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss   = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_dice += dice_score_batch(logits.detach(), masks)

    n = max(len(loader), 1)
    return total_loss / n, total_dice / n


@torch.no_grad()
def validate_one_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou  = 0.0

    for imgs, masks, _, _ in loader:
        imgs  = imgs.to(device)
        masks = masks.to(device)
        logits = model(imgs)
        loss   = loss_fn(logits, masks)
        total_loss += loss.item()
        total_dice += dice_score_batch(logits, masks)
        total_iou  += iou_score_batch(logits,  masks)

    n = max(len(loader), 1)
    return total_loss / n, total_dice / n, total_iou / n


# ─────────────────────────────────────────────────────────────────────────────
# Training Visualisation
# ─────────────────────────────────────────────────────────────────────────────
def plot_training_curves(history: dict, out_path: Path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="steelblue")
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   color="coral")
    axes[0].set_title("Training & Validation Loss");  axes[0].legend()
    axes[0].set_xlabel("Epoch");  axes[0].set_ylabel("Loss")

    axes[1].plot(epochs, history["train_dice"], label="Train Dice", color="steelblue")
    axes[1].plot(epochs, history["val_dice"],   label="Val Dice",   color="coral")
    axes[1].set_title("Dice Score");  axes[1].legend()
    axes[1].set_xlabel("Epoch");  axes[1].set_ylabel("Dice")

    axes[2].plot(epochs, history["val_iou"],  label="Val IoU",  color="seagreen")
    axes[2].plot(epochs, history["lr"],       label="LR × 1e3", color="purple",
                 linestyle="--")
    axes[2].set_title("Val IoU & Learning Rate");  axes[2].legend()
    axes[2].set_xlabel("Epoch")

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close(fig)
    logger.info("Training curves saved → %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Function
# ─────────────────────────────────────────────────────────────────────────────
def train(args):
    set_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    logger.info("Device: %s", device)

    # ── 1. Discover & split data ──────────────────────────────────────────
    pre_root = Path(args.preprocessed_root)
    df       = discover_slices(pre_root)
    if df.empty:
        logger.error("No preprocessed data found in %s", pre_root)
        return

    train_df, val_df, test_df = patient_split(
        df, train_frac=0.70, val_frac=0.15, seed=args.seed
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save splits for reproducibility
    train_df.to_csv(out_dir / "split_train.csv", index=False)
    val_df.to_csv(  out_dir / "split_val.csv",   index=False)
    test_df.to_csv( out_dir / "split_test.csv",  index=False)

    # ── 2. Datasets & Loaders ─────────────────────────────────────────────
    train_set = ProstateSliceDataset(train_df, augment=True)
    val_set   = ProstateSliceDataset(val_df,   augment=False)

    sampler = build_weighted_sampler(train_df)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size,
        sampler=sampler, num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
    )

    logger.info("Train slices: %d  |  Val slices: %d  |  Test slices: %d",
                len(train_set), len(val_set), len(test_df))

    # ── 3. Model ──────────────────────────────────────────────────────────
    model = build_unet(device=str(device))

    # ── 4. Optimiser, Loss, Scheduler ─────────────────────────────────────
    optimizer  = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn    = CombinedLoss(alpha=0.5)
    scheduler  = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                   patience=3)
    scaler     = (torch.cuda.amp.GradScaler()
                  if device.type == "cuda" and args.amp else None)

    # ── 5. Training loop ───────────────────────────────────────────────────
    best_val_loss  = float("inf")
    patience_count = 0
    ckpt_path      = Path(args.output_dir) / "models" / "combined_prostate_unet.pth"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    history = defaultdict(list)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_dice = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, scaler
        )
        vl_loss, vl_dice, vl_iou = validate_one_epoch(
            model, val_loader, loss_fn, device
        )
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(vl_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_dice"].append(tr_dice)
        history["val_dice"].append(vl_dice)
        history["val_iou"].append(vl_iou)
        history["lr"].append(current_lr * 1e3)   # scaled for plotting

        elapsed = time.time() - t0
        logger.info(
            "Epoch %02d/%02d | tr_loss=%.4f  vl_loss=%.4f  "
            "tr_dice=%.4f  vl_dice=%.4f  vl_iou=%.4f  lr=%.6f  [%.1fs]",
            epoch, args.epochs, tr_loss, vl_loss,
            tr_dice, vl_dice, vl_iou, current_lr, elapsed,
        )

        # ── Checkpoint saving ──────────────────────────────────────────────
        if vl_loss < best_val_loss:
            best_val_loss  = vl_loss
            patience_count = 0
            torch.save({
                "epoch"     : epoch,
                "model_state_dict"    : model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss"  : vl_loss,
                "val_dice"  : vl_dice,
                "history"   : dict(history),
            }, ckpt_path)
            logger.info("  ✓ Best model saved (val_loss=%.4f)", vl_loss)
        else:
            patience_count += 1
            logger.info("  · EarlyStopping patience %d/%d",
                        patience_count, args.patience)

        # ── Early stopping ─────────────────────────────────────────────────
        if patience_count >= args.patience:
            logger.info("Early stopping triggered at epoch %d.", epoch)
            break

    # ── 6. Save history & plots ────────────────────────────────────────────
    graphs_dir = out_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    plot_training_curves(history, graphs_dir / "training_curves.png")

    with open(out_dir / "training_history.json", "w") as f:
        json.dump(dict(history), f, indent=2)

    logger.info("\n✅ Training complete.")
    logger.info("   Best val loss : %.4f", best_val_loss)
    logger.info("   Checkpoint    : %s",   ckpt_path)
    logger.info("   Test set CSV  : %s",   out_dir / "split_test.csv")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train combined U-Net on all five prostate MRI datasets."
    )
    parser.add_argument("--preprocessed_root", type=str,
                        default="preprocessed",
                        help="Root folder containing preprocessed dataset subfolders")
    parser.add_argument("--output_dir",  type=str, default="results")
    parser.add_argument("--epochs",      type=int, default=30)
    parser.add_argument("--batch_size",  type=int, default=8)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--patience",    type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--amp",         action="store_true",
                        help="Use Automatic Mixed Precision (CUDA only)")
    args = parser.parse_args()
    train(args)

