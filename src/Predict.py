# src/predict.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ---------------------------
# Config
# ---------------------------
@dataclass
class CFG:
    project_root: Path = Path(__file__).resolve().parents[1]

    images_dir: Path = Path(__file__).resolve().parents[1] / "data" / "raw" / "images"
    splits_dir: Path = Path(__file__).resolve().parents[1] / "data" / "splits"

    out_dir: Path = Path(__file__).resolve().parents[1] / "outputs"
    preds_dir: Path = Path(__file__).resolve().parents[1] / "outputs" / "preds_test"
    visuals_dir: Path = Path(__file__).resolve().parents[1] / "outputs" / "visuals"

    img_size: int = 256
    batch_size: int = 1
    num_workers: int = 0

    threshold: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


cfg = CFG()


# ---------------------------
# Helpers
# ---------------------------
def read_split_txt(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def load_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# ---------------------------
# Dataset
# ---------------------------
class TestDataset(Dataset):
    def __init__(self, image_files: List[Path], size: int = 256):
        self.image_files = image_files
        self.size = size

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):
        p = self.image_files[idx]
        img = load_rgb(p)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img_t = torch.from_numpy(img).permute(2, 0, 1).float()
        return img_t, p.name


# ---------------------------
# Model (must match train.py)
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

        return self.head(d1)  # logits


# ---------------------------
# Visualization (optional but nice)
# ---------------------------
def save_overlay(rgb: np.ndarray, pred_mask_01: np.ndarray, out_path: Path) -> None:
    """
    rgb: HxWx3 RGB uint8
    pred_mask_01: HxW float/bool in {0,1}
    """
    h, w = pred_mask_01.shape
    overlay = rgb.copy()

    # make red-ish overlay where mask==1 (without hardcoding fancy styling)
    # We'll simply brighten the red channel a bit on roof pixels.
    roof = pred_mask_01.astype(bool)
    overlay[roof, 0] = np.clip(overlay[roof, 0] + 80, 0, 255)

    # side-by-side
    side = np.concatenate([rgb, overlay], axis=1)
    side_bgr = cv2.cvtColor(side, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), side_bgr)


# ---------------------------
# Predict
# ---------------------------
@torch.no_grad()
def run_predict(ckpt_path: Path, make_overlays: bool = True, max_overlays: int = 3) -> None:
    test_names = read_split_txt(cfg.splits_dir / "test.txt")
    test_files = [cfg.images_dir / n for n in test_names]

    for p in test_files:
        if not p.exists():
            raise FileNotFoundError(f"Missing test image: {p}")

    ds = TestDataset(test_files, size=cfg.img_size)
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
    )

    model = UNet(in_ch=3, out_ch=1, base=32).to(cfg.device)
    ckpt = torch.load(ckpt_path, map_location=cfg.device)

    # train.py saved dict with "model_state"
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    cfg.preds_dir.mkdir(parents=True, exist_ok=True)
    cfg.visuals_dir.mkdir(parents=True, exist_ok=True)

    overlay_count = 0

    for imgs, names in tqdm(loader, desc="predict"):
        imgs = imgs.to(cfg.device, non_blocking=True)
        logits = model(imgs)               # [B,1,H,W]
        probs = torch.sigmoid(logits)      # [B,1,H,W]

        for i in range(probs.size(0)):
            name = names[i]
            p = probs[i, 0].detach().cpu().numpy()  # HxW in [0,1]

            # binary mask
            mask01 = (p > cfg.threshold).astype(np.uint8)  # 0/1

            # save prediction as PNG (white roof, black background)
            out_mask = (mask01 * 255).astype(np.uint8)
            out_path = cfg.preds_dir / f"{Path(name).stem}_pred.png"
            cv2.imwrite(str(out_path), out_mask)

            # optional overlay for a few images (helps in README)
            if make_overlays and overlay_count < max_overlays:
                orig = load_rgb(cfg.images_dir / name)
                orig = cv2.resize(orig, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_AREA)
                vis_path = cfg.visuals_dir / f"{Path(name).stem}_overlay.png"
                save_overlay(orig.astype(np.uint8), mask01.astype(np.float32), vis_path)
                overlay_count += 1

    print(f"\n✅ Saved predictions to: {cfg.preds_dir}")
    if make_overlays:
        print(f"🖼 Saved example overlays to: {cfg.visuals_dir}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=str(cfg.out_dir / "models" / "best.pt"), help="Path to checkpoint")
    ap.add_argument("--no_overlays", action="store_true", help="Disable saving overlay visuals")
    ap.add_argument("--max_overlays", type=int, default=3, help="How many overlay images to save")
    return ap.parse_args()


def main():
    args = parse_args()
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    run_predict(
        ckpt_path=ckpt_path,
        make_overlays=(not args.no_overlays),
        max_overlays=args.max_overlays,
    )


if __name__ == "__main__":
    main()
