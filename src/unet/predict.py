"""Run a trained U-Net on droplet images and measure contact angles from its masks.

The substrate baseline is found with the same classical detector as v1.0, and
the angle fitting is the same code (`contact_angle.angles_from_mask`). Only the
segmentation step changes, so the two methods can be compared fairly.

Usage:
  python src/unet/predict.py                       # all images in data/raw
  python src/unet/predict.py --only A3_1_30
  python src/unet/predict.py --split val           # only held-out droplets
Outputs:
  outputs/unet_angles.csv          angles per image (+ classical + JayB columns)
  outputs/overlays_unet/*.jpg      QC overlays (same colour code as v1.0)
  data/masks_unet/*_unet_mask.png  full-resolution predicted masks
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from scipy import ndimage

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from common import build_model, pick_device, predict_prob  # noqa: E402
from contact_angle import angles_from_mask, deskew, detect_baseline, load_upright  # noqa: E402
from dataset import build_index  # noqa: E402


def load_model(ckpt_path: Path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ck["encoder"], pretrained=False)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), float(ck["scale"])


def clean_mask(prob: np.ndarray, yb: float, thr: float = 0.5) -> np.ndarray:
    """Threshold, keep the component that sits on the baseline, fill holes."""
    m = prob > thr
    m[int(round(yb)):] = False
    lab, n = ndimage.label(m)
    if n == 0:
        return m
    band = lab[max(int(yb) - 25, 0):int(yb)]
    ids = [i for i in np.unique(band) if i]
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    keep = max(ids, key=lambda i: sizes[i - 1]) if ids else int(np.argmax(sizes)) + 1
    return ndimage.binary_fill_holes(lab == keep)


def segment_unet(model, scale, img_d, yb, device, below=160):
    bottom = int(min(img_d.shape[0], round(yb) + below))
    crop = img_d[:bottom]
    small = cv2.resize(crop, (int(round(crop.shape[1] * scale)), int(round(bottom * scale))),
                       interpolation=cv2.INTER_AREA)
    prob_s = predict_prob(model, small, device)
    prob = cv2.resize(prob_s, (crop.shape[1], bottom), interpolation=cv2.INTER_LINEAR)
    full = np.zeros(img_d.shape[:2], np.float32)
    full[:bottom] = prob
    return clean_mask(full, yb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=ROOT / "models/unet_best.pt", type=Path)
    ap.add_argument("--root", default=ROOT / "data/raw", type=Path)
    ap.add_argument("--only", default=None)
    ap.add_argument("--split", choices=["train", "val"], default=None)
    a = ap.parse_args()

    device = pick_device()
    model, scale = load_model(a.model, device)
    idx = build_index(a.root)
    if a.only:
        idx = idx[idx.file.str.contains(a.only)]
    if a.split:
        meta = pd.read_csv(ROOT / "data/unet/meta.csv")
        idx = idx[idx.file.isin(meta.loc[meta.split == a.split, "file"])]

    rows = []
    for _, r in idx.iterrows():
        try:
            img = load_upright(r.path)
            base = detect_baseline(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            img_d, yb = deskew(img, base)
            mask = segment_unet(model, scale, img_d, yb, device)
            res = angles_from_mask(r.file, img_d, mask, yb, base.tilt_deg, base.support,
                                   out_dir=ROOT / "outputs/overlays_unet",
                                   mask_dir=ROOT / "data/masks_unet", suffix="_unet").row()
        except Exception as e:
            traceback.print_exc()
            res = {"file": r.file, "notes": f"FAILED: {e}"}
        rows.append({**r.to_dict(), **res})
        print(f"{r.file:28s} circle={res.get('theta_circle')!s:.5} "
              f"L={res.get('theta_left')!s:.5} R={res.get('theta_right')!s:.5} {res.get('notes', '')}")

    out = pd.DataFrame(rows)
    # side-by-side with the classical baseline and JayB's reference angles
    cls = ROOT / "outputs/baseline_angles.csv"
    if cls.exists():
        c = pd.read_csv(cls)[["file", "theta_circle", "theta_left", "theta_right"]]
        out = out.merge(c.add_suffix("_classical").rename(columns={"file_classical": "file"}),
                        on="file", how="left")
    ref = ROOT / "data/reference_jayb/jayb_angles.csv"
    if ref.exists():
        out = out.merge(pd.read_csv(ref), on="file", how="left")
    csv = ROOT / "outputs/unet_angles.csv"
    if (a.only or a.split) and csv.exists():
        old = pd.read_csv(csv)
        out = pd.concat([old[~old.file.isin(out.file)], out]).sort_values("file")
    out.to_csv(csv, index=False)
    print(f"wrote {csv} ({len(out)} rows)")


if __name__ == "__main__":
    main()
