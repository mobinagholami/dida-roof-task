# src/train.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ---------------------------
# Config
# ---------------------------
@dataclass
class CFG:
    # Paths
    project_root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = Path(__file__).resolve().parents[1] / "data" / "raw"
    images_dir: Path = Path(__file__).resolve().parents[1] / "data" / "raw" / "images"
    masks_dir: Path = Path(__file__).resolve().parents[1] / "data" / "raw" / "masks"
    splits_dir: Path = Path(__file__).resolve().parents[1] / "data" / "splits"
    out_dir: Path = Path(__file__).resolve().parents[1] / "outputs"
    models_dir: Path = Path(__file__).resolve().parents[1] / "outputs" / "models"

    # Training
    img_size: int = 256
    batch_size: int = 4
    lr: float = 1e-3
    epochs: int = 40
    weight_decay: float = 1e-4
    num_workers: int = 0

    # Repro
    seed: int = 42

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Loss weights
    dice_weight: float = 0.5  # final_loss = (1-dice_weight)*BCE + dice_weight*DiceLoss


cfg = CFG()


# ---------------------------
# Utils
# ---------------------------
def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_split_txt(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    return lines


def ensure_dirs() -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)


def load_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_mask_gray(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return m


# ---------------------------
# Dataset + simple augmentation
# ---------------------------
class RoofSegDataset(Dataset):
    def __init__(self, image_files: List[Path], mask_files: List[Path] | None, train: bool):
        self.image_files = image_files
        self.mask_files = mask_files
        self.train = train

    def __len__(self) -> int:
        return len(self.image_files)

    def _augment(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # light augmentations (good for tiny dataset)
        # horizontal flip
        if np.random.rand() < 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()

        # vertical flip
        if np.random.rand() < 0.5:
            img = np.flipud(img).copy()
            mask = np.flipud(mask).copy()

        # 90-degree rotations
        if np.random.rand() < 0.5:
            k = np.random.choice([1, 2, 3])
            img = np.rot90(img, k).copy()
            mask = np.rot90(mask, k).copy()

        return img, mask

    def __getitem__(self, idx: int):
        img = load_rgb(self.image_files[idx])
        img = cv2.resize(img, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img_t = torch.from_numpy(img).permute(2, 0, 1)  # CHW

        if self.mask_files is None:
            return img_t

        mask = load_mask_gray(self.mask_files[idx])
        mask = cv2.resize(mask, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_NEAREST)

        # binarize strictly: roof(white)=1, background(black)=0
        mask = (mask > 127).astype(np.float32)

        if self.train:
            img_aug, mask_aug = self._augment(img, mask)
            img_t = torch.from_numpy(img_aug).permute(2, 0, 1).float()
            mask_t = torch.from_numpy(mask_aug).unsqueeze(0).float()
        else:
            mask_t = torch.from_numpy(mask).unsqueeze(0).float()

        return img_t.float(), mask_t


# ---------------------------
# Model: a compact U-Net
# ---------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 1, base: int = 32):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base * 4, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        self.head = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        logits = self.head(d1)
        return logits


# ---------------------------
# Loss + Metrics
# ---------------------------
def dice_coeff_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean()


def iou_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    inter = (preds * targets).sum(dim=1)
    union = (preds + targets - preds * targets).sum(dim=1)
    iou = (inter + eps) / (union + eps)
    return iou.mean()


class DiceLoss(nn.Module):
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 1 - dice coefficient
        return 1.0 - dice_coeff_from_logits(logits, targets)


# ---------------------------
# Train / Eval
# ---------------------------
def make_loaders() -> Tuple[DataLoader, DataLoader]:
    train_names = read_split_txt(cfg.splits_dir / "train.txt")
    val_names = read_split_txt(cfg.splits_dir / "val.txt")

    train_imgs = [cfg.images_dir / n for n in train_names]
    train_masks = [cfg.masks_dir / n for n in train_names]

    val_imgs = [cfg.images_dir / n for n in val_names]
    val_masks = [cfg.masks_dir / n for n in val_names]

    # sanity checks
    for p in train_imgs + val_imgs:
        if not p.exists():
            raise FileNotFoundError(f"Missing image file: {p}")
    for p in train_masks + val_masks:
        if not p.exists():
            raise FileNotFoundError(f"Missing mask file: {p}")

    train_ds = RoofSegDataset(train_imgs, train_masks, train=True)
    val_ds = RoofSegDataset(val_imgs, val_masks, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
    )
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> Tuple[float, float, float]:
    model.eval()
    bce = 0.0
    dice = 0.0
    iou = 0.0
    n = 0

    for imgs, masks in loader:
        imgs = imgs.to(cfg.device, non_blocking=True)
        masks = masks.to(cfg.device, non_blocking=True)

        logits = model(imgs)
        loss_bce = F.binary_cross_entropy_with_logits(logits, masks).item()
        d = dice_coeff_from_logits(logits, masks).item()
        j = iou_from_logits(logits, masks).item()

        bs = imgs.size(0)
        bce += loss_bce * bs
        dice += d * bs
        iou += j * bs
        n += bs

    return bce / n, dice / n, iou / n


def train_one_epoch(model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer, scaler: torch.cuda.amp.GradScaler | None):
    model.train()
    bce_loss_fn = nn.BCEWithLogitsLoss()
    dice_loss_fn = DiceLoss()

    running = 0.0
    n = 0

    for imgs, masks in tqdm(loader, desc="train", leave=False):
        imgs = imgs.to(cfg.device, non_blocking=True)
        masks = masks.to(cfg.device, non_blocking=True)

        opt.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(imgs)
                bce = bce_loss_fn(logits, masks)
                dl = dice_loss_fn(logits, masks)
                loss = (1 - cfg.dice_weight) * bce + cfg.dice_weight * dl

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            logits = model(imgs)
            bce = bce_loss_fn(logits, masks)
            dl = dice_loss_fn(logits, masks)
            loss = (1 - cfg.dice_weight) * bce + cfg.dice_weight * dl

            loss.backward()
            opt.step()

        bs = imgs.size(0)
        running += loss.item() * bs
        n += bs

    return running / n


def main():
    set_seed(cfg.seed)
    ensure_dirs()

    print(f"Device: {cfg.device}")
    print(f"Images dir: {cfg.images_dir}")
    print(f"Masks dir:  {cfg.masks_dir}")

    train_loader, val_loader = make_loaders()

    model = UNet(in_ch=3, out_ch=1, base=32).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    use_amp = (cfg.device == "cuda")
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    best_val_dice = -1.0
    best_path = cfg.models_dir / "best.pt"

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, opt, scaler)
        val_bce, val_dice, val_iou = evaluate(model, val_loader)

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_bce={val_bce:.4f} val_dice={val_dice:.4f} val_iou={val_iou:.4f}"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "cfg": cfg.__dict__,
                    "epoch": epoch,
                    "val_dice": val_dice,
                },
                best_path,
            )
            print(f"  ✅ Saved new best model to: {best_path} (val_dice={val_dice:.4f})")

    print(f"Done. Best val dice: {best_val_dice:.4f}")
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
