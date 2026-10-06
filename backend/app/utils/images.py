"""
Image helpers: colormaps for heatmaps and base64 PNG encoding for the JSON API.

Kept dependency-light (NumPy + Pillow). A self-contained 'turbo'-style colormap is used so
heatmaps render without OpenCV/Matplotlib.
"""
from __future__ import annotations

import base64
import io

import numpy as np

try:
    from PIL import Image
    _PIL = True
except Exception:  # pragma: no cover - Pillow always present in the real backend
    _PIL = False


# A compact, perceptually-ordered colormap (approx. of Google's "turbo"): 8 anchor colors
# interpolated to 256 entries. Blue (low) -> green -> yellow -> red (high).
_TURBO_ANCHORS = np.array(
    [
        [48, 18, 59],
        [50, 121, 209],
        [30, 173, 158],
        [116, 205, 74],
        [227, 205, 46],
        [246, 143, 42],
        [216, 63, 34],
        [122, 4, 3],
    ],
    dtype=np.float64,
)


def _build_lut() -> np.ndarray:
    xs = np.linspace(0, 1, len(_TURBO_ANCHORS))
    grid = np.linspace(0, 1, 256)
    lut = np.stack([np.interp(grid, xs, _TURBO_ANCHORS[:, c]) for c in range(3)], axis=1)
    return lut.astype(np.uint8)


_LUT = _build_lut()


def normalize01(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    lo, hi = np.nanmin(a), np.nanmax(a)
    if hi - lo < 1e-9:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def apply_colormap(values: np.ndarray) -> np.ndarray:
    """Map a 2-D float array to an (H, W, 3) uint8 RGB heatmap via the turbo LUT."""
    idx = (normalize01(values) * 255).astype(np.uint8)
    return _LUT[idx]


def to_uint8_gray(image: np.ndarray) -> np.ndarray:
    return (normalize01(image) * 255).astype(np.uint8)


def resize_nearest(a: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Nearest-neighbour resize (works for 2-D or 3-D arrays) without external deps."""
    in_h, in_w = a.shape[:2]
    ys = (np.linspace(0, in_h - 1, out_h)).astype(int)
    xs = (np.linspace(0, in_w - 1, out_w)).astype(int)
    return a[ys][:, xs]


def overlay_heatmap(
    gray: np.ndarray,
    heat_values: np.ndarray,
    alpha: float = 0.45,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Alpha-blend a colormapped heatmap over a grayscale base image -> (H, W, 3) uint8.

    If `mask` (a bool array at base resolution) is given, the heatmap is only shown where
    the mask is True (e.g. inside the lung fields); elsewhere the plain grayscale shows
    through. This produces a clean, clinically-focused overlay.
    """
    base = to_uint8_gray(gray)
    base_rgb = np.stack([base] * 3, axis=2).astype(np.float64)
    heat = apply_colormap(resize_nearest(heat_values, base.shape[0], base.shape[1])).astype(np.float64)
    a = np.full(base.shape, alpha, dtype=np.float64)
    if mask is not None:
        a = a * (resize_nearest(mask.astype(np.uint8), base.shape[0], base.shape[1]) > 0)
    a = a[:, :, None]
    out = (1 - a) * base_rgb + a * heat
    return np.clip(out, 0, 255).astype(np.uint8)


def lung_overlay(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Tint the segmented lung fields (translucent cyan) over the grayscale image."""
    base = to_uint8_gray(gray)
    rgb = np.stack([base] * 3, axis=2).astype(np.float64)
    m = resize_nearest(mask.astype(np.uint8), base.shape[0], base.shape[1]) > 0
    tint = np.array([53, 224, 212], dtype=np.float64)  # cyan
    a = 0.28
    rgb[m] = (1 - a) * rgb[m] + a * tint
    # Draw a bright 1px-ish boundary by highlighting mask edges.
    edge = _mask_edge(m)
    rgb[edge] = np.array([120, 245, 235], dtype=np.float64)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _mask_edge(m: np.ndarray) -> np.ndarray:
    """Boolean edge of a mask via 1px shifts (no scipy dependency)."""
    e = np.zeros_like(m)
    e[:-1, :] |= m[:-1, :] != m[1:, :]
    e[:, :-1] |= m[:, :-1] != m[:, 1:]
    return e & m


def encode_png_base64(arr: np.ndarray) -> str:
    """Encode a uint8 array (H,W) or (H,W,3) as a data:image/png;base64 URI."""
    if not _PIL:
        raise RuntimeError("Pillow is required to encode PNGs")
    if arr.ndim == 2:
        img = Image.fromarray(arr.astype(np.uint8), mode="L")
    else:
        img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"
