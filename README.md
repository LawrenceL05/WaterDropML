# WaterDropML: contact-angle extraction (v1.0)

This project measures sessile-droplet contact angles automatically from Keyence
side-view microscope images (Fenning Research Group, UCSD).

v1.0 is a **classical computer-vision baseline**. It measures every image, and
the droplet masks it saves will be used to train a U-Net segmentation model in
the next version.

---

## 1. Workflow overview

```
 raw Keyence image (.jpg, substrate vertical on the right)
        │
        ▼
 ① Orient      rotate 90° clockwise → substrate horizontal, droplet on top
        │
        ▼
 ② Baseline    find the substrate line (thin bright ridge) with RANSAC,
                then deskew so it is perfectly horizontal.
                Everything below it is mirror reflection and is ignored.
        │
        ▼
 ③ Segment     half-resolution gradient map → strong-edge pixels
                → group sitting on the baseline → convex hull (first guess)
                → GrabCut refinement → droplet mask
        │
        ▼
 ④ Profile     row-by-row left/right edge + column-by-column top edge
        │
        ├──► ⑤a Circle fit   (spherical cap) → theta_circle   ← primary number
        └──► ⑤b Tangent fit  (quadratic near each contact point) → theta_left / theta_right
        │
        ▼
 ⑥ Outputs     outputs/baseline_angles.csv  (angles + metadata + QC notes)
                outputs/overlays/*.jpg       (visual check per image)
                data/masks_auto/*_mask.png   (pseudo-labels for U-Net)
                data/masks_auto/*_upright.jpg (matching rotated/deskewed images)
        │
        ▼
 ⑦ Compare     outputs/comparison_vs_jayb.csv (vs. JayB's ChatGPT-assisted angles)
```

### Why each step exists

| Step | Problem it solves |
|---|---|
| Orient | Raw frames are taken sideways. Rotating them means "up" is always away from the substrate. |
| Baseline + deskew | A contact angle is measured against the substrate, so the substrate must be found precisely. The sample is often tilted by 1–2°. |
| Ignore below baseline | The glass reflects the droplet. Without this, the reflection would be measured as part of the droplet. |
| Half-resolution gradients | JPEG noise at full resolution is as strong as the faint droplet rim, and downsampling suppresses it. |
| Convex hull → GrabCut | The hull is a robust first guess but cannot follow an undercut (θ > 90°). GrabCut refines the shape using colour. |
| Circle fit | It uses the whole profile, so it is insensitive to glare at the contact line. This makes it the most stable number. |
| Tangent fit | It gives separate left and right angles, which show asymmetry or pinning, but it is sensitive to glare. |

---

## 2. Repository layout

```
WaterDropML/
├── Contact angle/            # original images from JayB (incl. Test/ calibration shots)
├── discription.png           # screenshot of JayB's naming-convention message
├── data/
│   ├── raw/                  # the 59 measurement images (copied from Contact angle/)
│   ├── index.csv             # metadata parsed from filenames
│   ├── reference_jayb/       # JayB's annotated images + jayb_angles.csv
│   └── masks_auto/           # pseudo-label masks + upright images
├── src/
│   ├── dataset.py            # filename → metadata index
│   ├── contact_angle.py      # the classical pipeline
│   └── run_baseline.py       # batch runner
├── notebooks/
├── outputs/
│   ├── baseline_angles.csv
│   ├── comparison_vs_jayb.csv
│   └── overlays/
├── requirements.txt
└── README.md
```

`all_annotated_images.zip` and `Contact angle.zip` are not in git. The first is
over GitHub's 100 MB file limit, and both are just zipped copies of folders that
are already included.

### Filename convention (from JayB)

`YYYYMMDD_<A|B><t>_<droplet>_<mag>[_<shot>].jpg`

| Field | Meaning |
|---|---|
| A / B | A = no UVO treatment, B = UVO-treated |
| t | treatment time: 1 = 0 min, 2 = 30 min, 3 = 1 h, 4 = 3 h |
| droplet | droplet 1 or 2 |
| mag | magnification: 20x, 30x or 50x |
| shot | shot sequence number (optional) |

---

## 3. Step-by-step: recreating v1.0

### 3.1 Install

```bash
git clone <this repo URL>
cd WaterDropML
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.10, OpenCV 4.x, SciPy 1.15, scikit-image 0.25 and pandas 2.

### 3.2 Put the images in place

The images are already in `data/raw/` in this repo. For a new batch, copy the
`.jpg` files into `data/raw/` and keep the filename convention above. Leave out
calibration and test shots.

### 3.3 Build the metadata index

```bash
python src/dataset.py data/raw
```

This writes `data/index.csv`, with one row per image giving the condition, time,
droplet, magnification and shot. It prints a count per group so you can check
nothing is missing. The current set has 59 images.

### 3.4 Run the measurement

```bash
python src/run_baseline.py                 # all images (~3 s per image)
python src/run_baseline.py --only A3_1_30  # just one image or a group (updates rows in place)
```

### 3.5 Check the results (do not skip this)

Open `outputs/overlays/<name>_overlay.jpg` for each image:

| Colour | Meaning |
|---|---|
| cyan line | detected substrate baseline |
| yellow dots | detected droplet profile |
| magenta circle | spherical-cap fit |
| red lines + numbers | left/right tangent angles |
| orange dots | points used for the tangent fit |

Accept a measurement only if the cyan line sits on the substrate and the yellow
dots follow the droplet's outer edge. Also check the `notes` column in
`outputs/baseline_angles.csv`:

- `left/right contact out of frame`: only one side can be measured, and the circle is fitted to a partial arc.
- `L/R disagree >15 deg`: glare at a contact point. Use `theta_circle`.
- `FAILED`: no droplet was found. This happens on very flat UVO droplets.

### 3.6 Compare with JayB's reference angles

```python
import pandas as pd
a = pd.read_csv("outputs/baseline_angles.csv")
j = pd.read_csv("data/reference_jayb/jayb_angles.csv")
d = a.merge(j, on="file")
d["diff"] = d.theta_circle - d.theta_jayb
d.to_csv("outputs/comparison_vs_jayb.csv", index=False)
print(d.groupby("condition")["diff"].describe())
```

### 3.7 Use it on a single image from Python

```python
import sys; sys.path.insert(0, "src")
from contact_angle import measure
r = measure("data/raw/20261015_A3_1_30.jpg", out_dir="outputs/overlays", mask_dir="data/masks_auto")
print(r.theta_circle, r.theta_left, r.theta_right, r.notes)
```

### 3.8 Settings you can tune (in `src/contact_angle.py`)

| Function | Parameter | Effect |
|---|---|---|
| `detect_baseline` | `y_range` | Vertical band (fraction of height) searched for the substrate line |
| `segment_droplet` | `scale` | Working resolution. Lower is less noisy but coarser. |
| `tangent_angle` | `band_frac` | Height of the fit band above the contact line (default 25%) |
| `tangent_angle` | `margin_frac` | Rows skipped right at the contact line to avoid glare (default 4%) |

---

## 4. v1.0 results (2026-11-06)

- **A3_1_30:** circle fit 75.4°, left tangent 73.4°, right tangent 79.8°. JayB's estimate was 73.0°.
- **Agreement with JayB:** the median absolute difference is about 5.6° on the full untreated (A) 30x droplets and about 4° on the full UVO-treated (B) droplets. Neither set has been checked against a reference instrument yet.
- **Untreated (A) droplets:** about 75–80°.
- **UVO-treated (B) droplets:** about 5–20°, which is much more hydrophilic.

### Known limitations

- The tangent angles change by about 5–15° with the fit-band settings because of glare at the contact line. Use `theta_circle` as the primary number.
- Most 50x frames cut off one contact point.
- Some UVO droplets are wider than the frame at 20x and 30x. These give meaningless circle values (about 1–3° or 180°).
- 2 images (B2_1_30_1, B2_2_50_2) fail to segment.

---

## 5. Roadmap

1. **v1.1:** review the masks in `data/masks_auto/` and fix bad ones by hand (CVAT or Label Studio).
2. **v2.0:** train a U-Net (`segmentation-models-pytorch`) on the upright images and corrected masks, then put its masks through the same angle-fitting code.
3. **Data:** build a water, acetone and IPA dataset on a dedicated contact-angle rig, as Eric suggested.
4. **Validation:** measure a reference sample with known contact angles.
