"""Build the U-Net training set from upright images + masks.

For every measured image:
  * the label is the hand-corrected mask (data/masks_gt/<stem>_mask.png) if it
    exists, otherwise the classical pseudo-label (data/masks_auto/<stem>_mask.png)
  * the upright, deskewed image is cropped from the top of the frame down to
    `--below` px under the substrate baseline (so the reflection is included as
    background the model must learn to ignore), then downscaled by `--scale`
  * images are split into train / val by DROPLET (sample_id), never by shot

Excluded automatically:
  * images the classical run FAILED on (no mask)
  * images whose classical circle angle is implausible (<4 or >170 deg) -> the
    pseudo-label is almost certainly wrong (flat droplet wider than frame)
  * images marked status=bad in data/qc_review.csv (if that file exists)
  (an image with a masks_gt label is always kept)

Output (data/unet/):
  images/<stem>.png   masks/<stem>.png (0/255)   meta.csv

Usage:
  python src/unet/prepare_data.py
  python src/unet/prepare_data.py --val A3_1 B4_1 A2_2 --scale 0.25
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--angles", default=ROOT / "outputs/baseline_angles.csv", type=Path)
    ap.add_argument("--auto", default=ROOT / "data/masks_auto", type=Path)
    ap.add_argument("--gt", default=ROOT / "data/masks_gt", type=Path)
    ap.add_argument("--qc", default=ROOT / "data/qc_review.csv", type=Path)
    ap.add_argument("--out", default=ROOT / "data/unet", type=Path)
    ap.add_argument("--scale", type=float, default=0.25)
    ap.add_argument("--below", type=int, default=160, help="px kept below the baseline")
    ap.add_argument("--val", nargs="+", default=["A3_1", "B4_1", "A2_2"],
                    help="sample_ids (droplets) held out for validation")
    a = ap.parse_args()

    df = pd.read_csv(a.angles)
    qc_bad = set()
    if a.qc.exists():
        qc = pd.read_csv(a.qc)
        qc_bad = set(qc.loc[qc.status.str.strip().str.lower() == "bad", "file"])

    (a.out / "images").mkdir(parents=True, exist_ok=True)
    (a.out / "masks").mkdir(parents=True, exist_ok=True)
    rows, skipped = [], []
    for _, r in df.iterrows():
        stem = Path(r.file).stem
        gt = a.gt / f"{stem}_mask.png"
        auto = a.auto / f"{stem}_mask.png"
        up = a.auto / f"{stem}_upright.jpg"
        if gt.exists():
            src, label = gt, "gt"
        else:
            notes = str(r.get("notes", "") or "")
            th = r.get("theta_circle")
            if "FAILED" in notes or not auto.exists():
                skipped.append((r.file, "no classical mask")); continue
            if pd.isna(th) or not (4 <= th <= 170):
                skipped.append((r.file, f"implausible circle angle {th}")); continue
            if r.file in qc_bad:
                skipped.append((r.file, "qc_review: bad")); continue
            src, label = auto, "auto"
        if not up.exists():
            skipped.append((r.file, "missing upright image")); continue

        img = cv2.imread(str(up))
        m = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
        full_h, full_w = img.shape[:2]
        yb = float(r.y_base)
        bottom = int(min(img.shape[0], round(yb) + a.below))
        img, m = img[:bottom], m[:bottom]
        size = (int(round(img.shape[1] * a.scale)), int(round(bottom * a.scale)))
        img_s = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        m_s = (cv2.resize(m, size, interpolation=cv2.INTER_AREA) > 127).astype(np.uint8) * 255
        cv2.imwrite(str(a.out / "images" / f"{stem}.png"), img_s)
        cv2.imwrite(str(a.out / "masks" / f"{stem}.png"), m_s)
        rows.append({
            "file": r.file, "stem": stem, "sample_id": r.sample_id,
            "condition": r.condition, "magnification": r.magnification,
            "split": "val" if r.sample_id in a.val else "train",
            "label_source": label, "scale": a.scale, "crop_bottom": bottom,
            "y_base": yb, "full_w": full_w, "full_h": full_h,
        })
    meta = pd.DataFrame(rows)
    meta.to_csv(a.out / "meta.csv", index=False)
    print(meta.groupby(["split", "label_source"]).size().to_string())
    print(f"{len(meta)} images written to {a.out}; {len(skipped)} skipped:")
    for f, why in skipped:
        print(f"  {f}: {why}")


if __name__ == "__main__":
    main()
