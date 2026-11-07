"""Compare classical and U-Net masks against hand-corrected ground truth.

For every image that has a hand-corrected mask in data/masks_gt/, this scores
  * mask overlap (IoU, Dice) and outline distance in full-resolution pixels
  * angle error: angles measured from each mask minus angles from the GT mask
for both the classical masks (data/masks_auto/) and the U-Net masks
(data/masks_unet/, from predict.py).

Only images on HELD-OUT droplets give a fair score for the U-Net; the `split`
column says which images were used for training.

Usage:
  python src/unet/evaluate.py
Output:
  outputs/eval_vs_gt.csv  and a summary printed to the console
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from common import boundary_error, dice_iou  # noqa: E402
from contact_angle import angles_from_mask  # noqa: E402


def read_mask(p: Path):
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return None if m is None else m > 127


def main():
    gt_dir = ROOT / "data/masks_gt"
    gts = sorted(gt_dir.glob("*_mask.png"))
    if not gts:
        sys.exit("no hand-corrected masks in data/masks_gt/ yet - see docs/UNET_GUIDE.md step 3")
    base = pd.read_csv(ROOT / "outputs/baseline_angles.csv").set_index("file")
    meta_p = ROOT / "data/unet/meta.csv"
    split = pd.read_csv(meta_p).set_index("stem")["split"] if meta_p.exists() else pd.Series(dtype=str)

    rows = []
    for g in gts:
        stem = g.name[: -len("_mask.png")]
        file = stem + ".jpg"
        img = cv2.imread(str(ROOT / "data/masks_auto" / f"{stem}_upright.jpg"))
        if img is None or file not in base.index:
            continue
        yb = float(base.loc[file, "y_base"])
        gt = read_mask(g)
        ref = angles_from_mask(file, img, gt, yb)
        row = {"file": file, "split": split.get(stem, "n/a"),
               "theta_circle_gt": ref.theta_circle,
               "theta_left_gt": ref.theta_left, "theta_right_gt": ref.theta_right}
        for name, p in (("classical", ROOT / "data/masks_auto" / f"{stem}_mask.png"),
                        ("unet", ROOT / "data/masks_unet" / f"{stem}_unet_mask.png")):
            m = read_mask(p)
            if m is None:
                continue
            d, iou = dice_iou(m, gt)
            row[f"{name}_iou"], row[f"{name}_dice"] = iou, d
            row[f"{name}_boundary_px"] = boundary_error(m, gt)
            try:
                r = angles_from_mask(file, img, m, yb)
                for k in ("circle", "left", "right"):
                    v, t = getattr(r, f"theta_{k}"), row[f"theta_{k}_gt"]
                    row[f"{name}_err_{k}"] = (v - t) if (v is not None and t is not None) else np.nan
            except Exception:
                pass
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "outputs/eval_vs_gt.csv", index=False)
    cols = [c for c in df.columns if c.endswith(("_iou", "_boundary_px")) or "_err_" in c]
    summ = df.groupby("split")[cols].agg(lambda s: np.nanmedian(np.abs(s)))
    print("median |value| per split (errors in degrees, boundary in full-res px):")
    print(summ.round(2).T.to_string())
    print(f"wrote outputs/eval_vs_gt.csv ({len(df)} images)")


if __name__ == "__main__":
    main()
