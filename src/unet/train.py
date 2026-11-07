"""Train a U-Net to segment the droplet (above the substrate) in side-view images.

Inputs come from `src/unet/prepare_data.py` (data/unet/). Training uses random
crops biased towards the droplet outline, plus simple photometric/geometric
augmentation. Validation runs on whole held-out images (split by droplet).

Usage:
  python src/unet/train.py                         # defaults (GPU if available)
  python src/unet/train.py --epochs 3 --crop 256   # quick smoke test on CPU
Outputs:
  models/unet_best.pt        best checkpoint by validation IoU (+ config)
  models/train_log.csv       per-epoch loss and validation metrics
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (boundary_error, build_model, dice_iou, pick_device,  # noqa: E402
                    predict_prob, to_tensor)

ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_pair(data_dir: Path, stem: str):
    img = cv2.imread(str(data_dir / "images" / f"{stem}.png"))
    m = cv2.imread(str(data_dir / "masks" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE) > 127
    return img, m


def augment(img: np.ndarray, m: np.ndarray, rng: random.Random):
    """Light augmentation. No vertical flip: droplets always sit on top."""
    if rng.random() < 0.5:                                     # mirror left/right
        img, m = img[:, ::-1], m[:, ::-1]
    h, w = m.shape
    if rng.random() < 0.7:                                     # small rotate / scale
        ang = rng.uniform(-3, 3)
        sc = rng.uniform(0.9, 1.1)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
        img = cv2.warpAffine(np.ascontiguousarray(img), M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
        m = cv2.warpAffine(np.ascontiguousarray(m.astype(np.uint8)), M, (w, h),
                           flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT) > 0
    img = img.astype(np.float32)
    if rng.random() < 0.8:                                     # brightness / contrast
        img = img * rng.uniform(0.75, 1.25) + rng.uniform(-25, 25)
    if rng.random() < 0.3:                                     # colour cast
        img = img * np.array([rng.uniform(0.9, 1.1) for _ in range(3)], np.float32)
    if rng.random() < 0.3:                                     # defocus
        k = rng.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    if rng.random() < 0.3:                                     # sensor / JPEG noise
        img = img + np.random.normal(0, rng.uniform(2, 6), img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8), m


class CropDataset(Dataset):
    """Random crops; `edge_frac` of them are centred on a point of the outline."""

    def __init__(self, data_dir, stems, crop=384, samples_per_image=16, edge_frac=0.7, seed=0):
        self.items = [load_pair(data_dir, s) for s in stems]
        self.edges = []
        for _, m in self.items:
            mu = m.astype(np.uint8)
            e = (mu - cv2.erode(mu, np.ones((3, 3), np.uint8))) > 0
            self.edges.append(np.argwhere(e))
        self.crop, self.n, self.edge_frac = crop, samples_per_image, edge_frac
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.items) * self.n

    def __getitem__(self, i):
        img, m = self.items[i // self.n]
        edges = self.edges[i // self.n]
        h, w = m.shape
        c = min(self.crop, h, w)
        if len(edges) and self.rng.random() < self.edge_frac:
            cy, cx = edges[self.rng.randrange(len(edges))]
            cy += self.rng.randint(-c // 4, c // 4)
            cx += self.rng.randint(-c // 4, c // 4)
        else:
            cy, cx = self.rng.randrange(h), self.rng.randrange(w)
        y0 = int(np.clip(cy - c // 2, 0, h - c))
        x0 = int(np.clip(cx - c // 2, 0, w - c))
        ci, cm = augment(img[y0:y0 + c, x0:x0 + c], m[y0:y0 + c, x0:x0 + c], self.rng)
        return to_tensor(ci), torch.from_numpy(cm.astype(np.float32))[None]


# --------------------------------------------------------------------------
# Loss / evaluation
# --------------------------------------------------------------------------
def dice_bce_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum((2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((2, 3)) + target.sum((2, 3)) + 1)
    return bce + dice.mean()


def evaluate(model, data_dir, stems, device, scale):
    rows = []
    for s in stems:
        img, gt = load_pair(data_dir, s)
        pred = predict_prob(model, img, device) > 0.5
        d, iou = dice_iou(pred, gt)
        be = boundary_error(pred, gt) / scale  # in full-resolution pixels
        rows.append({"stem": s, "dice": d, "iou": iou, "boundary_px_full": be})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=ROOT / "data/unet", type=Path)
    ap.add_argument("--out", default=ROOT / "models", type=Path)
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=384)
    ap.add_argument("--samples-per-image", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gt-only", action="store_true",
                    help="train only on hand-corrected masks (label_source == gt)")
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    meta = pd.read_csv(a.data / "meta.csv")
    train_meta = meta[meta.split == "train"]
    if a.gt_only:
        train_meta = train_meta[train_meta.label_source == "gt"]
    val_stems = list(meta.loc[meta.split == "val", "stem"])
    if train_meta.empty or not val_stems:
        sys.exit("need at least one train and one val image - check data/unet/meta.csv")
    scale = float(meta.scale.iloc[0])
    print(f"train images: {len(train_meta)} "
          f"({(train_meta.label_source == 'gt').sum()} hand-corrected), val images: {len(val_stems)}")

    device = pick_device()
    print("device:", device)
    model = build_model(a.encoder, pretrained=not a.no_pretrained).to(device)
    ds = CropDataset(a.data, list(train_meta.stem), a.crop, a.samples_per_image, seed=a.seed)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = max(1, a.epochs * len(dl))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))

    a.out.mkdir(parents=True, exist_ok=True)
    log, best = [], -1.0
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            loss = dice_bce_loss(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item()
        ev = evaluate(model, a.data, val_stems, device, scale)
        row = {"epoch": ep, "train_loss": tot / max(1, len(dl)),
               "val_dice": ev.dice.mean(), "val_iou": ev.iou.mean(),
               "val_boundary_px_full": ev.boundary_px_full.mean(), "sec": time.time() - t0}
        log.append(row)
        pd.DataFrame(log).to_csv(a.out / "train_log.csv", index=False)
        flag = ""
        if row["val_iou"] > best:
            best = row["val_iou"]
            torch.save({"state_dict": model.state_dict(), "encoder": a.encoder,
                        "scale": scale, "epoch": ep, "val": row}, a.out / "unet_best.pt")
            flag = "  <- best"
        print(f"ep {ep:3d}  loss {row['train_loss']:.4f}  val IoU {row['val_iou']:.4f}  "
              f"Dice {row['val_dice']:.4f}  boundary {row['val_boundary_px_full']:.1f}px  "
              f"({row['sec']:.0f}s){flag}")
    print(f"best val IoU {best:.4f} -> {a.out / 'unet_best.pt'}")


if __name__ == "__main__":
    main()
