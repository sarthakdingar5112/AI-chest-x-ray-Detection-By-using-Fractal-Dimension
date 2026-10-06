"""
Input validation: is the uploaded image actually a chest X-ray?

Two cheap, complementary signals:
  1. Color saturation — real X-rays are grayscale. A color photo/screenshot has high
     saturation and is rejected outright (works even without the ML stack).
  2. Lung detection — when the segmentation model is available, a genuine chest X-ray yields
     a sizeable lung mask. If essentially no lung is found, it's not a chest X-ray (this
     catches grayscale non-X-rays like scanned text or greyscale photos).

If neither signal fires, the image is accepted. Thresholds are deliberately lenient to avoid
rejecting unusual-but-valid X-rays.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np

try:
    from PIL import Image
    _PIL = True
except Exception:
    _PIL = False

# A color photo averages ~0.3-0.6 saturation; grayscale X-rays ~0. 0.15 leaves wide margin.
COLOR_SATURATION_MAX = 0.15
# Real chest X-rays: lungs are ~20-40% of the frame. Reject only when almost none is found.
MIN_LUNG_FRACTION = 0.015

COLOR_MESSAGE = (
    "This looks like a color image, not a grayscale X-ray. "
    "Please upload a chest X-ray image (PNG, JPG or DICOM)."
)
NO_LUNG_MESSAGE = (
    "No lung fields could be detected, so this does not appear to be a chest X-ray. "
    "Please upload a chest X-ray image."
)


class NotAnXrayError(Exception):
    """Raised when the uploaded image does not appear to be a chest X-ray."""


@dataclass
class XrayAssessment:
    is_xray: bool
    reason: str
    saturation: float
    lung_fraction: float


def mean_saturation(data: bytes, filename: str) -> float:
    """Average HSV-style saturation of the image (0 = grayscale, higher = more colorful)."""
    name = (filename or "").lower()
    if name.endswith((".dcm", ".dicom")) or (len(data) > 132 and data[128:132] == b"DICM"):
        return 0.0  # DICOM is inherently grayscale
    if not _PIL:
        return 0.0
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return 0.0
    img.thumbnail((256, 256))  # small sample is enough and fast
    arr = np.asarray(img, dtype=np.float64)
    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    sat = np.divide(mx - mn, mx, out=np.zeros_like(mx), where=mx > 0)
    return float(sat.mean())


def assess(saturation: float, seg_available: bool, lung_fraction: float) -> XrayAssessment:
    """Combine the signals into an accept/reject decision."""
    if saturation > COLOR_SATURATION_MAX:
        return XrayAssessment(False, COLOR_MESSAGE, saturation, lung_fraction)
    if seg_available and lung_fraction < MIN_LUNG_FRACTION:
        return XrayAssessment(False, NO_LUNG_MESSAGE, saturation, lung_fraction)
    return XrayAssessment(True, "ok", saturation, lung_fraction)
