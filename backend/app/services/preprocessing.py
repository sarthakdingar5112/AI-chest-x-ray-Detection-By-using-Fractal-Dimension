"""
Image loading & preprocessing for chest X-rays.

Handles PNG/JPG and DICOM input, converts to grayscale, applies CLAHE contrast
enhancement (standard for X-ray), and produces both a display-sized grayscale image (for
fractal analysis & overlays) and a 224x224 tensor-ready array for the CNN.

Heavy deps (cv2, pydicom) are imported lazily so the fractal core can be used without them.
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


@dataclass
class PreprocessedImage:
    gray_full: np.ndarray      # float [0,1], display resolution (for fractal + overlays)
    cnn_input: np.ndarray      # float, 224x224, normalized to TorchXRayVision range
    original_uint8: np.ndarray # uint8 grayscale for display encoding


def _load_any(data: bytes, filename: str) -> np.ndarray:
    """Return a 2-D float grayscale array from raw bytes (PNG/JPG or DICOM)."""
    name = (filename or "").lower()
    if name.endswith(".dcm") or name.endswith(".dicom"):
        return _load_dicom(data)
    # Try DICOM by magic number even without extension, else fall back to PIL.
    if data[128:132] == b"DICM":
        return _load_dicom(data)
    if not _PIL:
        raise RuntimeError("Pillow is required to read PNG/JPG images")
    img = Image.open(io.BytesIO(data)).convert("L")
    return np.asarray(img, dtype=np.float64)


def _load_dicom(data: bytes) -> np.ndarray:
    import pydicom  # lazy
    ds = pydicom.dcmread(io.BytesIO(data), force=True)
    arr = ds.pixel_array.astype(np.float64)
    # Apply MONOCHROME1 inversion if present so bone is bright.
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr
    return arr


def _clahe(gray_uint8: np.ndarray) -> np.ndarray:
    """Contrast-Limited Adaptive Histogram Equalization. Uses cv2 if available."""
    try:
        import cv2  # lazy
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray_uint8)
    except Exception:
        # Fallback: global histogram equalization via numpy.
        hist, bins = np.histogram(gray_uint8.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_masked = np.ma.masked_equal(cdf, 0)
        cdf_masked = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())
        cdf_final = np.ma.filled(cdf_masked, 0).astype(np.uint8)
        return cdf_final[gray_uint8]


def preprocess(data: bytes, filename: str, display_size: int = 512) -> PreprocessedImage:
    gray = _load_any(data, filename)

    # Normalize to [0,255] uint8 for CLAHE.
    g = gray - gray.min()
    if g.max() > 0:
        g = g / g.max()
    g8 = (g * 255).astype(np.uint8)
    enhanced = _clahe(g8)

    # Display / fractal image: resize longest side to display_size.
    disp = _resize(enhanced, display_size)
    gray_full = disp.astype(np.float64) / 255.0

    # CNN input: 224x224 normalized to TorchXRayVision's expected [-1024, 1024] range.
    cnn = _resize_square(enhanced, 224).astype(np.float64)
    cnn = (cnn / 255.0) * 2048.0 - 1024.0  # [0,255] -> [-1024,1024]

    return PreprocessedImage(
        gray_full=gray_full,
        cnn_input=cnn,
        original_uint8=disp.astype(np.uint8),
    )


def _resize(gray8: np.ndarray, longest: int) -> np.ndarray:
    h, w = gray8.shape
    scale = longest / max(h, w)
    if scale >= 1:
        return gray8
    out_h, out_w = int(round(h * scale)), int(round(w * scale))
    return _pil_resize(gray8, out_w, out_h)


def _resize_square(gray8: np.ndarray, size: int) -> np.ndarray:
    return _pil_resize(gray8, size, size)


def _pil_resize(gray8: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    if _PIL:
        img = Image.fromarray(gray8).resize((out_w, out_h), Image.BILINEAR)
        return np.asarray(img)
    # Nearest fallback.
    ys = np.linspace(0, gray8.shape[0] - 1, out_h).astype(int)
    xs = np.linspace(0, gray8.shape[1] - 1, out_w).astype(int)
    return gray8[ys][:, xs]
