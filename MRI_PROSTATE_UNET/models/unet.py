"""
=============================================================================
unet.py  –  Standard U-Net for Prostate MRI Segmentation
=============================================================================

Architecture
------------
  Encoder:    4 levels, channels [32, 64, 128, 256]
  Bottleneck: 512 channels
  Decoder:    4 levels with skip connections
  Output:     1 channel (binary prostate mask)

  Input shape : (B, 1, 256, 256)
  Output shape: (B, 1, 256, 256)  – logits (apply sigmoid for probabilities)

Reference
---------
  Ronneberger O., Fischer P., Brox T. (2015). U-Net: Convolutional Networks
  for Biomedical Image Segmentation. MICCAI 2015.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary as torchinfo_summary


# ─────────────────────────────────────────────────────────────────────────────
# Building Blocks
# ─────────────────────────────────────────────────────────────────────────────
class DoubleConv(nn.Module):
    """
    Two consecutive:  Conv2d → BatchNorm → ReLU
    Used in every encoder/decoder stage and the bottleneck.
    """
    def __init__(self, in_ch: int, out_ch: int, mid_ch: int | None = None):
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """DoubleConv followed by MaxPool2d."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv    = DoubleConv(in_ch, out_ch)
        self.pool    = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        skip = self.conv(x)    # used as skip connection
        out  = self.pool(skip)
        return out, skip


class DecoderBlock(nn.Module):
    """TransposedConv (up-sampling) → concatenate skip → DoubleConv."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch,
                                        kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch * 2, out_ch)   # concat doubles channels

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # Handle odd spatial dimensions
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ─────────────────────────────────────────────────────────────────────────────
# U-Net
# ─────────────────────────────────────────────────────────────────────────────
class UNet(nn.Module):
    """
    Standard U-Net for binary prostate segmentation.

    Parameters
    ----------
    in_channels      : int   Number of input channels (1 for grayscale MRI)
    out_channels     : int   Number of output channels (1 for binary mask)
    encoder_channels : list  Feature map sizes at each encoder level
    bottleneck_ch    : int   Feature map size at bottleneck
    """

    def __init__(
        self,
        in_channels     : int       = 1,
        out_channels    : int       = 1,
        encoder_channels: list[int] = None,
        bottleneck_ch   : int       = 512,
    ):
        super().__init__()

        if encoder_channels is None:
            encoder_channels = [32, 64, 128, 256]

        assert len(encoder_channels) == 4, "Exactly 4 encoder levels required."

        # ── Encoder ────────────────────────────────────────────────────────
        self.enc1 = EncoderBlock(in_channels,        encoder_channels[0])
        self.enc2 = EncoderBlock(encoder_channels[0], encoder_channels[1])
        self.enc3 = EncoderBlock(encoder_channels[1], encoder_channels[2])
        self.enc4 = EncoderBlock(encoder_channels[2], encoder_channels[3])

        # ── Bottleneck ──────────────────────────────────────────────────────
        self.bottleneck = DoubleConv(encoder_channels[3], bottleneck_ch)

        # ── Decoder ─────────────────────────────────────────────────────────
        self.dec4 = DecoderBlock(bottleneck_ch,       encoder_channels[3])
        self.dec3 = DecoderBlock(encoder_channels[3], encoder_channels[2])
        self.dec2 = DecoderBlock(encoder_channels[2], encoder_channels[1])
        self.dec1 = DecoderBlock(encoder_channels[1], encoder_channels[0])

        # ── Output head ──────────────────────────────────────────────────────
        self.output_conv = nn.Conv2d(encoder_channels[0], out_channels,
                                     kernel_size=1)

    # ── Forward ────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 1, H, W)   Normalised MRI image

        Returns
        -------
        logits : (B, 1, H, W)   Raw scores (apply sigmoid externally)
        """
        # Encoder path
        x,   s1 = self.enc1(x)   # s1: (B, 32,  H/1,  W/1)  → pool → x: /2
        x,   s2 = self.enc2(x)   # s2: (B, 64,  H/2,  W/2)
        x,   s3 = self.enc3(x)   # s3: (B, 128, H/4,  W/4)
        x,   s4 = self.enc4(x)   # s4: (B, 256, H/8,  W/8)

        # Bottleneck
        x = self.bottleneck(x)   # (B, 512, H/16, W/16)

        # Decoder path
        x = self.dec4(x, s4)     # (B, 256, H/8,  W/8)
        x = self.dec3(x, s3)     # (B, 128, H/4,  W/4)
        x = self.dec2(x, s2)     # (B, 64,  H/2,  W/2)
        x = self.dec1(x, s1)     # (B, 32,  H,    W)

        return self.output_conv(x)   # (B, 1, H, W)

    # ── Helpers ────────────────────────────────────────────────────────────
    def predict(self, x: torch.Tensor,
                threshold: float = 0.5) -> torch.Tensor:
        """Apply sigmoid and threshold to produce a binary mask."""
        with torch.no_grad():
            logits = self.forward(x)
            probs  = torch.sigmoid(logits)
            return (probs > threshold).float()

    def count_parameters(self) -> dict:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ─────────────────────────────────────────────────────────────────────────────
# Loss Functions
# ─────────────────────────────────────────────────────────────────────────────
class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.
    Works on logits (applies sigmoid internally).
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs   = probs.view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        return 1.0 - (2.0 * intersection + self.smooth) / \
                     (probs.sum() + targets.sum() + self.smooth)


class CombinedLoss(nn.Module):
    """
    Dice Loss + Binary Cross-Entropy Loss.
    alpha * Dice + (1 - alpha) * BCE
    """
    def __init__(self, alpha: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.alpha    = alpha
        self.dice     = DiceLoss(smooth=smooth)
        self.bce      = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice(logits, targets)
        bce_loss  = self.bce(logits, targets)
        return self.alpha * dice_loss + (1.0 - self.alpha) * bce_loss


# ─────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ─────────────────────────────────────────────────────────────────────────────
def build_unet(in_channels: int = 1,
               out_channels: int = 1,
               encoder_channels: list | None = None,
               bottleneck_ch: int = 512,
               device: str | None = None) -> UNet:
    """
    Build and return a U-Net model, optionally moved to a device.

    Example
    -------
    >>> model = build_unet()
    >>> x = torch.randn(2, 1, 256, 256)
    >>> out = model(x)
    >>> print(out.shape)   # (2, 1, 256, 256)
    """
    if encoder_channels is None:
        encoder_channels = [32, 64, 128, 256]

    model = UNet(in_channels=in_channels,
                 out_channels=out_channels,
                 encoder_channels=encoder_channels,
                 bottleneck_ch=bottleneck_ch)

    if device is not None:
        model = model.to(device)

    params = model.count_parameters()
    print(f"\n{'='*55}")
    print(f"  U-Net Architecture Summary")
    print(f"{'='*55}")
    print(f"  Input  channels : {in_channels}")
    print(f"  Output channels : {out_channels}")
    print(f"  Encoder stages  : {encoder_channels}")
    print(f"  Bottleneck      : {bottleneck_ch}")
    print(f"  Total params    : {params['total']:,}")
    print(f"  Trainable params: {params['trainable']:,}")
    print(f"{'='*55}\n")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = build_unet(device=device)

    # Forward pass
    x   = torch.randn(2, 1, 256, 256).to(device)
    out = model(x)
    print(f"Input  : {x.shape}")
    print(f"Output : {out.shape}")

    # Loss
    y    = torch.randint(0, 2, (2, 1, 256, 256)).float().to(device)
    loss_fn = CombinedLoss(alpha=0.5)
    loss = loss_fn(out, y)
    print(f"Combined Loss: {loss.item():.4f}")

    # Prediction
    pred = model.predict(x)
    print(f"Prediction : {pred.shape}, values: {pred.unique()}")

    # torchinfo summary
    try:
        torchinfo_summary(model, input_size=(2, 1, 256, 256), device=device)
    except Exception:
        print("(torchinfo not installed – skipping detailed summary)")

