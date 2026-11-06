"""Build a metadata index from the droplet image filenames.

Naming convention (from JayB, 2026-10-16):
    YYYYMMDD_<cond><time>_<droplet>_<mag>[_<shot>].jpg
    cond : A = no UVO treatment, B = UVO treatment
    time : 1 = 0 min, 2 = 30 min, 3 = 1 h, 4 = 3 h
    droplet : 1 or 2
    mag  : 20 / 30 / 50  (x magnification)
    shot : optional shot sequence number
Files in the Test/ sub-folder are excluded.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

TIME_MIN = {1: 0, 2: 30, 3: 60, 4: 180}
PATTERN = re.compile(r"^(\d{8})_([AB])(\d)_(\d)_(\d+)(?:_(\d+))?$")


def parse_name(stem: str) -> dict | None:
    m = PATTERN.match(stem)
    if not m:
        return None
    date, cond, t, drop, mag, shot = m.groups()
    return {
        "date": date,
        "condition": cond,
        "uvo": cond == "B",
        "time_code": int(t),
        "time_min": TIME_MIN[int(t)],
        "droplet": int(drop),
        "magnification": int(mag),
        "shot": int(shot) if shot else 1,
        "sample_id": f"{cond}{t}_{drop}",
    }


def build_index(root: str | Path) -> pd.DataFrame:
    rows = []
    for p in sorted(Path(root).rglob("*.jpg")):
        if "test" in (part.lower() for part in p.parts) or "__MACOSX" in p.parts:
            continue
        meta = parse_name(p.stem)
        if meta is None:
            continue
        rows.append({"file": p.name, "path": str(p), **meta})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    df = build_index(root)
    out = Path("data/index.csv")
    df.to_csv(out, index=False)
    print(df.groupby(["condition", "time_min", "magnification"]).size())
    print(f"{len(df)} images -> {out}")
