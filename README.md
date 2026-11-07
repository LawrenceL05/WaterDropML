# WaterDropML: contact-angle extraction (v1.0)

This project measures sessile-droplet contact angles automatically from Keyence
side-view microscope images (Fenning Research Group, UCSD).

v1.0 is a **classical computer-vision baseline**. It measures every image, and
the droplet masks it saves will be used to train a U-Net segmentation model in
the next version.

---

## 1. Workflow overview

![WaterDropML workflow from Keyence image to circle and tangent contact-angle estimates](docs/diagrams/01-measurement-workflow.svg)

[Editable Excalidraw source](docs/diagrams/01-measurement-workflow.excalidraw)

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

![WaterDropML repository map showing inputs, Python scripts, and generated results](docs/diagrams/02-repository-map.svg)

[Editable Excalidraw source](docs/diagrams/02-repository-map.excalidraw)

`all_annotated_images.zip` and `Contact angle.zip` are not in git. The first is
over GitHub's 100 MB file limit, and both are just zipped copies of folders that
are already included.

### Filename convention (from JayB)

![Annotated example of a droplet filename with date, condition and time, droplet, magnification, and shot fields](docs/diagrams/03-filename-key.svg)

[Editable Excalidraw source](docs/diagrams/03-filename-key.excalidraw)

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

![Contact-angle geometry and review checklist, with example values from A3_1_30](docs/diagrams/04-contact-angle-and-qc.svg)

[Editable Excalidraw source](docs/diagrams/04-contact-angle-and-qc.excalidraw)

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

The comparison uses JayB's ChatGPT-assisted estimates as a reference.

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
