"""Shared helpers for U-Net training and inference."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(encoder: str = "resnet34", pretrained: bool = True):
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name=encoder,
                    encoder_weights="imagenet" if pretrained else None,
                    in_channels=3, classes=1)


def to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    """uint8 BGR HxWx3 -> normalised float 3xHxW (RGB, ImageNet stats)."""
    rgb = img_bgr[:, :, ::-1].astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))


def pad_to_multiple(x: torch.Tensor, m: int = 32):
    """Pad a (...,H,W) tensor on the bottom/right so H and W are multiples of m."""
    h, w = x.shape[-2:]
    ph, pw = (-h) % m, (-w) % m
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect" if min(h, w) > max(ph, pw) else "constant")
    return x, (h, w)


@torch.no_grad()
def predict_prob(model, img_bgr: np.ndarray, device) -> np.ndarray:
    """Full-image probability map (H x W, float32 in [0,1]). Uses h-flip TTA."""
    model.eval()
    x, (h, w) = pad_to_multiple(to_tensor(img_bgr)[None].to(device))
    p = torch.sigmoid(model(x))
    p = (p + torch.flip(torch.sigmoid(model(torch.flip(x, dims=[-1]))), dims=[-1])) / 2
    return p[0, 0, :h, :w].float().cpu().numpy()


def dice_iou(pred: np.ndarray, gt: np.ndarray):
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = (pred & gt).sum()
    s = pred.sum() + gt.sum()
    union = (pred | gt).sum()
    dice = 2 * inter / s if s else 1.0
    iou = inter / union if union else 1.0
    return float(dice), float(iou)


def boundary_error(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean symmetric distance (px) between the outlines of two masks."""
    import cv2
    from scipy import ndimage

    def edge(m):
        m = m.astype(np.uint8)
        return (m - cv2.erode(m, np.ones((3, 3), np.uint8))) > 0

    ep, eg = edge(pred), edge(gt)
    if not ep.any() or not eg.any():
        return float("nan")
    dg = ndimage.distance_transform_edt(~eg)
    dp = ndimage.distance_transform_edt(~ep)
    return float((dg[ep].mean() + dp[eg].mean()) / 2)
