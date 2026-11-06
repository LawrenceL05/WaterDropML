"""Run the classical contact-angle baseline on every indexed image.

Usage (from the project root):
    python src/run_baseline.py                # all images
    python src/run_baseline.py --only A3_1_30 # filter by substring
Outputs:
    outputs/baseline_angles.csv   per-image angles + metadata
    outputs/overlays/*.jpg        visual check of every measurement
    data/masks_auto/*_mask.png    pseudo-label masks for U-Net training
    data/masks_auto/*_upright.jpg matching rotated/deskewed images
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from contact_angle import measure  # noqa: E402
from dataset import build_index  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()

    idx = build_index(a.root)
    if a.only:
        idx = idx[idx.file.str.contains(a.only)]
    rows = []
    for _, r in idx.iterrows():
        try:
            res = measure(r.path, "outputs/overlays", "data/masks_auto").row()
        except Exception as e:  # keep going, record the failure
            res = {"file": r.file, "notes": f"FAILED: {e}"}
            traceback.print_exc()
        rows.append({**r.to_dict(), **res})
        print(f"{r.file:28s} L={res.get('theta_left')!s:.5} R={res.get('theta_right')!s:.5} "
              f"circle={res.get('theta_circle')!s:.5} {res.get('notes', '')}")
    out = pd.DataFrame(rows)
    Path("outputs").mkdir(exist_ok=True)
    csv = Path("outputs/baseline_angles.csv")
    if a.only and csv.exists():  # update only the re-run rows
        old = pd.read_csv(csv)
        out = pd.concat([old[~old.file.isin(out.file)], out]).sort_values("file")
    out.to_csv(csv, index=False)
    print(f"wrote outputs/baseline_angles.csv ({len(out)} rows)")


if __name__ == "__main__":
    main()
