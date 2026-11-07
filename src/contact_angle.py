"""Classical (non-ML) contact-angle extraction for side-view droplet images.

Pipeline
--------
1. Rotate the raw Keyence image so the substrate is horizontal and the
   droplet sits on top (raw images have the substrate on the right).
2. Detect the substrate baseline: a thin bright ridge running across the
   whole frame. Fit it robustly (RANSAC) and deskew the image so the
   baseline becomes perfectly horizontal.
3. Segment the droplet above the baseline (gradient-seeded convex hull,
   refined with GrabCut) and take the outer profile row by row (left/right)
   and column by column (top).
   Pixels below the baseline are ignored (mirror reflection).
4. Measure the angle:
     * circle fit (spherical-cap assumption, one angle for both sides)
     * local polynomial tangent at each contact point (left / right)
5. Save an overlay and a binary mask (the mask doubles as a pseudo-label
   for training the U-Net later).

All coordinates below are in the rotated + deskewed frame, y pointing down.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage, optimize


# --------------------------------------------------------------------------
# I/O and orientation
# --------------------------------------------------------------------------
def load_upright(path: str | Path) -> np.ndarray:
    """Load a raw image and rotate so the droplet sits on a horizontal substrate."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    if img.shape[1] > img.shape[0]:  # raw landscape frames: substrate on the right
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


# --------------------------------------------------------------------------
# Baseline detection
# --------------------------------------------------------------------------
@dataclass
class Baseline:
    slope: float      # dy/dx in the rotated (not yet deskewed) frame
    intercept: float  # y at x = 0
    support: float    # fraction of column strips agreeing with the line

    def y_at(self, x):
        return self.slope * np.asarray(x) + self.intercept

    @property
    def tilt_deg(self) -> float:
        return math.degrees(math.atan(self.slope))


def detect_baseline(gray: np.ndarray, strip: int = 40, n_peaks: int = 4,
                    max_tilt_deg: float = 5.0, tol: float = 6.0,
                    y_range: tuple[float, float] = (0.25, 0.85)) -> Baseline:
    """Find the substrate line as the top-most horizontal ridge seen across the frame.

    For every vertical strip we keep the strongest bright-ridge rows
    (negative 2nd derivative). Lines are then proposed from pairs of
    candidates in different strips, and a line is accepted if many strips
    agree. Among accepted lines the top-most one is the substrate surface
    (lower lines are the glass bottom / stage edges).
    """
    h, w = gray.shape
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 2)
    ridge = -cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=5)
    y0, y1 = int(h * y_range[0]), int(h * y_range[1])

    xs, ys, ws = [], [], []
    for x0 in range(0, w - strip, strip):
        prof = ridge[y0:y1, x0:x0 + strip].mean(1)
        prof = ndimage.maximum_filter1d(prof, 15)
        order = np.argsort(prof)[::-1]
        picked = []
        for i in order:
            if prof[i] <= 0:
                break
            if all(abs(i - p) > 15 for p in picked):
                picked.append(i)
            if len(picked) == n_peaks:
                break
        # refine to the true local max in the unfiltered profile
        raw = ridge[y0:y1, x0:x0 + strip].mean(1)
        for p in picked:
            lo, hi = max(p - 8, 0), min(p + 9, len(raw))
            yy = lo + int(np.argmax(raw[lo:hi]))
            xs.append(x0 + strip / 2)
            ys.append(y0 + yy)
            ws.append(raw[yy])
    xs, ys, ws = map(np.asarray, (xs, ys, ws))
    n_strips = (w - strip) // strip

    max_slope = math.tan(math.radians(max_tilt_deg))
    rng = np.random.default_rng(0)
    lines = []
    for _ in range(4000):
        i, j = rng.choice(len(xs), 2, replace=False)
        if abs(xs[i] - xs[j]) < w * 0.25:
            continue
        m = (ys[j] - ys[i]) / (xs[j] - xs[i])
        if abs(m) > max_slope:
            continue
        b = ys[i] - m * xs[i]
        inl = np.abs(ys - (m * xs + b)) < tol
        support = len(np.unique(xs[inl])) / n_strips
        lines.append((support, m, b, inl))
    if not lines:
        raise RuntimeError("no baseline candidates")
    best_support = max(l[0] for l in lines)
    good = [l for l in lines if l[0] >= 0.8 * best_support]
    # top-most line at the image centre
    support, m, b, inl = min(good, key=lambda l: l[1] * w / 2 + l[2])
    # refine with weighted least squares on inliers
    A = np.vstack([xs[inl], np.ones(inl.sum())]).T
    m, b = np.linalg.lstsq(A * ws[inl, None], ys[inl] * ws[inl], rcond=None)[0]
    return Baseline(float(m), float(b), float(support))


def deskew(img: np.ndarray, base: Baseline) -> tuple[np.ndarray, float]:
    """Rotate so the baseline is horizontal. Returns image and baseline row."""
    h, w = img.shape[:2]
    cx = w / 2
    cy = float(base.y_at(cx))
    M = cv2.getRotationMatrix2D((cx, cy), base.tilt_deg, 1.0)
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
    return out, cy


# --------------------------------------------------------------------------
# Droplet segmentation
# --------------------------------------------------------------------------
def gradient_map(img: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Edge strength combining luminance and chroma (droplet rim is greenish)."""
    lab = cv2.cvtColor(cv2.GaussianBlur(img, (0, 0), sigma), cv2.COLOR_BGR2LAB).astype(np.float32)
    G = np.zeros(img.shape[:2], np.float32)
    for c, wgt in ((0, 1.0), (1, 2.0), (2, 2.0)):
        gx = cv2.Sobel(lab[:, :, c], cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(lab[:, :, c], cv2.CV_32F, 0, 1, ksize=3)
        G += wgt * (gx * gx + gy * gy)
    return np.sqrt(G)


def segment_droplet(img: np.ndarray, y_base: float, scale: float = 0.5,
                    gc_iters: int = 4) -> np.ndarray:
    """Boolean droplet mask (full resolution) above the baseline.

    1. Work at reduced resolution (less JPEG noise). Strong-gradient pixels
       (threshold from the empty top of the frame) are grouped; the group
       sitting on the baseline is the droplet. Its convex hull is a good
       first guess of the cap.
    2. GrabCut refines the hull using colour models, so non-convex shapes
       (theta > 90 deg undercut) are recovered. Result is clipped to the
       slightly dilated hull so glare/background cannot leak in.
    """
    yb = int(round(y_base))
    crop = img[:yb]
    sm = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = sm.shape[:2]
    G = gradient_map(sm, 1.5)
    bg = G[: max(int(0.1 * h), 10)]
    T = float(np.percentile(bg, 99.5)) * 1.5
    strong = G > T
    strong[-2:] = False
    lab_img, n = ndimage.label(ndimage.binary_dilation(strong, iterations=6))
    if n == 0:
        raise RuntimeError("no droplet edges found")
    ids = set(np.unique(lab_img[h - 15:h])) - {0}
    if not ids:
        raise RuntimeError("nothing touches the baseline")
    sizes = ndimage.sum(strong, lab_img, range(1, n + 1))
    best = max(ids, key=lambda i: sizes[i - 1])
    grp = (lab_img == best) & strong
    ys, xs = np.nonzero(grp)
    bot = ys > h - 15
    pts = np.vstack([np.c_[xs, ys], [[xs[bot].min(), h - 1], [xs[bot].max(), h - 1]]]).astype(np.int32)
    hull = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(pts), 1)

    m = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    m[hull > 0] = cv2.GC_PR_FGD
    m[ndimage.binary_erosion(hull, iterations=15)] = cv2.GC_FGD
    m[: max(int(0.05 * h), 5)] = cv2.GC_BGD
    m[~ndimage.binary_dilation(hull, iterations=60)] = cv2.GC_BGD
    bgm = np.zeros((1, 65), np.float64)
    fgm = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(sm, m, None, bgm, fgm, gc_iters, cv2.GC_INIT_WITH_MASK)
        fg = (m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD)
    except cv2.error:
        fg = hull > 0
    fg &= ndimage.binary_dilation(hull, iterations=3)
    fg = ndimage.binary_fill_holes(ndimage.binary_opening(fg, iterations=2))
    lab2, n2 = ndimage.label(fg)
    if n2 == 0:
        fg = hull > 0
    else:
        s2 = ndimage.sum(fg, lab2, range(1, n2 + 1))
        fg = lab2 == (int(np.argmax(s2)) + 1)
    full = cv2.resize(fg.astype(np.uint8) * 255, (img.shape[1], yb),
                      interpolation=cv2.INTER_LINEAR) > 127
    out = np.zeros(img.shape[:2], bool)
    out[:yb] = full
    return out


@dataclass
class Profile:
    y: np.ndarray       # rows (ascending)
    xl: np.ndarray      # left-most droplet edge per row (nan if none)
    xr: np.ndarray      # right-most droplet edge per row
    x: np.ndarray       # columns
    ytop: np.ndarray    # top-most droplet edge per column (nan if none)


def extract_profile(mask: np.ndarray, y_base: float) -> Profile:
    h, w = mask.shape
    yb = int(round(y_base))
    rows = mask[:yb]
    has = rows.any(1)
    y = np.arange(yb, dtype=float)
    xl = np.where(has, rows.argmax(1), np.nan).astype(float)
    xr = np.where(has, w - 1 - rows[:, ::-1].argmax(1), np.nan).astype(float)
    hasc = rows.any(0)
    x = np.arange(w, dtype=float)
    ytop = np.where(hasc, rows.argmax(0), np.nan).astype(float)
    # light de-spiking
    for arr in (xl, xr):
        ok = ~np.isnan(arr)
        if ok.sum() > 7:
            arr[ok] = ndimage.median_filter(arr[ok], 5)
    ok = ~np.isnan(ytop)
    if ok.sum() > 7:
        ytop[ok] = ndimage.median_filter(ytop[ok], 5)
    return Profile(y, xl, xr, x, ytop)


# --------------------------------------------------------------------------
# Angle measurement
# --------------------------------------------------------------------------
def fit_circle(pts: np.ndarray):
    x, y = pts[:, 0], pts[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    cx, cy, c = np.linalg.lstsq(A, x ** 2 + y ** 2, rcond=None)[0]
    r = math.sqrt(max(c + cx ** 2 + cy ** 2, 1.0))

    def res(q):
        return np.hypot(x - q[0], y - q[1]) - q[2]
    sol = optimize.least_squares(res, [cx, cy, r], loss="soft_l1", f_scale=2.0)
    return sol.x  # cx, cy, r


def circle_angle(cx, cy, r, y_base) -> float:
    d = y_base - cy  # >0 if centre above baseline
    return math.degrees(math.pi / 2 + math.asin(float(np.clip(d / r, -1, 1))))


def _robust_polyfit(u, v, deg):
    p = np.polyfit(u, v, deg)
    for _ in range(2):
        r = v - np.polyval(p, u)
        s = 1.4826 * np.median(np.abs(r)) + 0.5
        keep = np.abs(r) < 3 * s
        if keep.sum() < deg + 3:
            break
        p = np.polyfit(u[keep], v[keep], deg)
    return p


def tangent_angle(p: Profile, y_base: float, side: str, width: int,
                  band_frac: float = 0.25, min_band: float = 25,
                  margin_frac: float = 0.04, min_margin: float = 6):
    """Contact angle from a local polynomial fit near one contact point.

    Steep side (theta > ~35 deg): fit x = f(y) on the row-wise edge over a
    band above the baseline. Shallow side: fit y = g(x) on the top profile.
    Returns (theta_deg, x_contact, fit_points) or (None, None, None).
    """
    xe = p.xl if side == "left" else p.xr
    ok = ~np.isnan(xe)
    if ok.sum() < 10:
        return None, None, None
    height = y_base - p.y[ok].min()
    band = max(height * band_frac, min_band)
    # skip the few rows right at the contact line: glare / reflection blur
    margin = max(height * margin_frac, min_margin)
    sel = ok & (p.y > y_base - band) & (p.y < y_base - margin)
    sel &= (xe > 3) & (xe < width - 4)
    th = None
    if sel.sum() >= 8:
        u, v = p.y[sel], xe[sel]
        q = _robust_polyfit(u, v, 2 if sel.sum() > 20 else 1)
        xc = float(np.polyval(q, y_base))
        d = float(np.polyval(np.polyder(q), y_base))
        th = math.atan2(1, -d) if side == "left" else math.atan2(1, d)
        pts = np.c_[v, u]
    if th is None or math.degrees(th) < 35:
        # shallow drop: use the top profile next to the contact point
        okc = ~np.isnan(p.ytop)
        if okc.sum() < 10:
            return (None, None, None) if th is None else (math.degrees(th), xc, pts)
        xs_all = p.x[okc]
        xedge = xs_all.min() if side == "left" else xs_all.max()
        if xedge <= 3 or xedge >= width - 4:
            return None, None, None
        span = xs_all.max() - xs_all.min()
        bx = max(span * band_frac, 2 * min_band)
        if side == "left":
            s2 = okc & (p.x >= xedge) & (p.x < xedge + bx)
        else:
            s2 = okc & (p.x <= xedge) & (p.x > xedge - bx)
        s2 &= p.ytop < y_base - margin / 2
        if s2.sum() < 8:
            return None, None, None
        u, v = p.x[s2], p.ytop[s2]
        q = _robust_polyfit(u, v, 2 if s2.sum() > 20 else 1)
        roots = np.roots(np.polysub(q, [y_base]))
        roots = roots[np.isreal(roots)].real
        xc = float(roots[np.argmin(np.abs(roots - xedge))]) if len(roots) else float(xedge)
        dydx = float(np.polyval(np.polyder(q), xc))
        th = math.atan(-dydx) if side == "left" else math.atan(dydx)
        if th < 0:
            th += math.pi
        pts = np.c_[u, v]
    return math.degrees(th), xc, pts


@dataclass
class Result:
    file: str
    baseline_tilt_deg: float
    baseline_support: float
    y_base: float
    height_px: float
    base_width_px: float | None
    theta_circle: float | None
    theta_left: float | None
    theta_right: float | None
    left_in_frame: bool
    right_in_frame: bool
    notes: list[str] = field(default_factory=list)

    def row(self):
        d = asdict(self)
        d["notes"] = "; ".join(self.notes)
        return d


def measure(path: str | Path, out_dir: str | Path | None = None,
            mask_dir: str | Path | None = None) -> Result:
    path = Path(path)
    img = load_upright(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    base = detect_baseline(gray)
    img_d, yb = deskew(img, base)
    mask = segment_droplet(img_d, yb)
    return angles_from_mask(path.name, img_d, mask, yb, base.tilt_deg, base.support,
                            out_dir=out_dir, mask_dir=mask_dir)


def angles_from_mask(name: str, img_d: np.ndarray, mask: np.ndarray, yb: float,
                     tilt_deg: float = 0.0, support: float = float("nan"),
                     out_dir: str | Path | None = None,
                     mask_dir: str | Path | None = None,
                     suffix: str = "") -> Result:
    """Measure contact angles from a droplet mask on an upright, deskewed image.

    Shared by the classical pipeline and the U-Net: whatever produced the mask,
    the baseline handling and angle fitting are identical, so the two can be
    compared fairly.
    """
    stem = Path(name).stem
    mask = np.asarray(mask).astype(bool).copy()
    mask[int(round(yb)):] = False
    h, w = img_d.shape[:2]
    if not mask.any():
        raise RuntimeError("empty mask")
    prof = extract_profile(mask, yb)
    notes = []

    ys, xs = np.nonzero(mask)
    height = float(yb - ys.min())
    left_in = bool(xs.min() > 5)
    right_in = bool(xs.max() < w - 6)
    if not left_in:
        notes.append("left contact out of frame")
    if not right_in:
        notes.append("right contact out of frame")
    if ys.min() <= 3:
        notes.append("droplet top cut by frame")

    # circle fit on all outer-profile points away from the baseline and frame
    pts = []
    for xe, inside in ((prof.xl, left_in), (prof.xr, right_in)):
        if not inside:  # that side is the frame-cut boundary, not the droplet
            continue
        ok = ~np.isnan(xe) & (prof.y < yb - 6) & (xe > 3) & (xe < w - 4)
        pts.append(np.c_[xe[ok], prof.y[ok]])
    okc = ~np.isnan(prof.ytop) & (prof.ytop > 3) & (prof.ytop < yb - 6)
    okc &= (prof.x > 8) & (prof.x < w - 9)
    pts.append(np.c_[prof.x[okc], prof.ytop[okc]])
    pts = np.unique(np.vstack(pts).round(1), axis=0)
    th_c = circ = None
    if len(pts) > 50:
        circ = fit_circle(pts)
        th_c = circle_angle(*circ, yb)
        if not (left_in and right_in):
            notes.append("circle fitted to partial arc")

    th_l, xl, fl = tangent_angle(prof, yb, "left", w) if left_in else (None, None, None)
    th_r, xr, fr = tangent_angle(prof, yb, "right", w) if right_in else (None, None, None)
    width = (xr - xl) if (xl is not None and xr is not None) else None
    if th_l is not None and th_r is not None and abs(th_l - th_r) > 15:
        notes.append("L/R disagree >15 deg")

    res = Result(name, tilt_deg, support, yb, height, width,
                 th_c, th_l, th_r, left_in, right_in, notes)

    if mask_dir is not None:
        Path(mask_dir).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(Path(mask_dir) / (stem + suffix + "_mask.png")), mask.astype(np.uint8) * 255)
        if not suffix:
            cv2.imwrite(str(Path(mask_dir) / (stem + "_upright.jpg")), img_d)
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        draw_overlay(img_d, pts, yb, res, circ, (xl, fl), (xr, fr),
                     Path(out_dir) / (stem + suffix + "_overlay.jpg"))
    return res


def draw_overlay(img, pts, yb, res, circ, left, right, out_path):
    o = img.copy()
    h, w = o.shape[:2]
    cv2.line(o, (0, int(yb)), (w, int(yb)), (255, 200, 0), 2)
    if circ is not None:
        cx, cy, r = circ
        if r < 5 * max(h, w):
            cv2.circle(o, (int(cx), int(cy)), int(r), (255, 0, 255), 1)
    for x, y in pts[::2]:
        cv2.circle(o, (int(x), int(y)), 1, (0, 255, 255), -1)
    for side, (xc, fp), th in (("L", left, res.theta_left), ("R", right, res.theta_right)):
        if xc is None or th is None:
            continue
        for x, y in fp[::2]:
            cv2.circle(o, (int(x), int(y)), 2, (0, 140, 255), -1)
        t = math.radians(th)
        L = 300
        dx = math.cos(t) * L * (1 if side == "L" else -1)
        dy = -math.sin(t) * L
        cv2.line(o, (int(xc), int(yb)), (int(xc + dx), int(yb + dy)), (0, 0, 255), 3)
        tx = int(xc) - 190 if side == "L" else int(xc) + 20
        cv2.putText(o, f"{th:.1f}", (tx, int(yb) - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                    (0, 0, 255), 4)
    title = (f"{res.file}  L={_f(res.theta_left)}  R={_f(res.theta_right)}  "
             f"circle={_f(res.theta_circle)}")
    cv2.rectangle(o, (0, 0), (w, 70), (20, 20, 20), -1)
    cv2.putText(o, title, (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.imwrite(str(out_path), o)


def _f(v):
    return "n/a" if v is None else f"{v:.1f}"


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--out", default="outputs/overlays")
    ap.add_argument("--masks", default="data/masks_auto")
    a = ap.parse_args()
    for p in a.images:
        r = measure(p, a.out, a.masks)
        print(r.row())
