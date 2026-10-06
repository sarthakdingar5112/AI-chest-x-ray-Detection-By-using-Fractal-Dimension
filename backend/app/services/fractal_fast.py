"""
Vectorised (fast) versions of the fractal measures in ``fractal.py``.

Why this file exists
--------------------
``fractal.py`` is written for clarity (explicit loops, easy to read in a viva). That is too
slow to run over a whole *dataset* (1000+ images). This module computes the SAME quantities
with NumPy reshaping tricks, ~50-100x faster, so we can do dataset-level statistics.

``tests/test_fractal_fast.py`` checks that these agree with the readable implementations.

All functions take a 2-D array. For speed the image is cropped to a size divisible by the
largest box (power of two), exactly as the box-counting theory assumes.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def _pow2_sizes(lo: int, hi: int) -> List[int]:
    out, s = [], lo
    while s <= hi:
        out.append(s)
        s *= 2
    return out


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])


def _crop_to_multiple(a: np.ndarray, m: int) -> np.ndarray:
    h, w = a.shape
    return a[: (h // m) * m, : (w // m) * m]


def _block_reduce(a: np.ndarray, r: int, fn) -> np.ndarray:
    h, w = a.shape
    return fn(a.reshape(h // r, r, w // r, r), axis=(1, 3))


# --------------------------------------------------------------------------------------
# Box counting (binary)
# --------------------------------------------------------------------------------------
def box_counting_fd(binary: np.ndarray, min_box: int = 2, max_box: int | None = None):
    b = np.asarray(binary) != 0
    h, w = b.shape
    if max_box is None:
        max_box = max(min_box, 2 ** int(np.floor(np.log2(min(h, w) / 2))))
    sizes = _pow2_sizes(min_box, max_box)
    b = _crop_to_multiple(b, max_box)
    counts = []
    for s in sizes:
        counts.append(max(int(_block_reduce(b, s, np.any).sum()), 1))
    fd = -_slope(np.log(sizes), np.log(counts))
    return fd, list(zip(sizes, counts))


# --------------------------------------------------------------------------------------
# Differential box counting (grayscale)
# --------------------------------------------------------------------------------------
def dbc_fd(image: np.ndarray, min_box: int = 2, max_box: int | None = None):
    g = np.asarray(image, dtype=np.float64)
    gmin, gmax = g.min(), g.max()
    if gmax - gmin < 1e-9:
        return float("nan"), []
    G = 255.0
    g = (g - gmin) / (gmax - gmin) * G
    h, w = g.shape
    W = min(h, w)
    if max_box is None:
        max_box = max(min_box, 2 ** int(np.floor(np.log2(W / 2))))
    sizes = _pow2_sizes(min_box, max_box)
    g = _crop_to_multiple(g, max_box)
    counts = []
    for r in sizes:
        h_r = max(1.0, G * r / W)
        k_max = np.floor(_block_reduce(g, r, np.max) / h_r)
        k_min = np.floor(_block_reduce(g, r, np.min) / h_r)
        counts.append(max(int((k_max - k_min + 1).sum()), 1))
    fd = _slope(np.log(1.0 / np.array(sizes)), np.log(counts))
    return fd, list(zip(sizes, counts))


# --------------------------------------------------------------------------------------
# Lacunarity (gliding box via integral image, fully vectorised)
# --------------------------------------------------------------------------------------
def lacunarity(binary: np.ndarray, box: int = 16) -> float:
    b = (np.asarray(binary) != 0).astype(np.float64)
    h, w = b.shape
    if box >= min(h, w):
        box = max(1, min(h, w) // 2)
    ii = np.pad(b.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    m = ii[box:, box:] - ii[:-box, box:] - ii[box:, :-box] + ii[:-box, :-box]
    mean = m.mean()
    if mean <= 0:
        return float("nan")
    return float((m ** 2).mean() / mean ** 2)


def lacunarity_curve(binary, sizes=(4, 8, 16, 32, 64)):
    return [(s, lacunarity(binary, s)) for s in sizes]


# --------------------------------------------------------------------------------------
# Multifractal spectrum width (method of moments, reshape-based box masses)
# --------------------------------------------------------------------------------------
def multifractal(image: np.ndarray, q_values=None, min_box: int = 2, max_box: int | None = None):
    g = np.asarray(image, dtype=np.float64)
    g = g - g.min()
    if g.max() <= 0:
        return [], [], float("nan")
    if q_values is None:
        q_values = np.linspace(-5, 5, 21)
    h, w = g.shape
    W = min(h, w)
    if max_box is None:
        max_box = max(min_box, 2 ** int(np.floor(np.log2(W / 2))))
    scales = _pow2_sizes(min_box, max_box)
    gc = _crop_to_multiple(g, max_box)
    log_r = np.log(np.array(scales, dtype=np.float64))

    P = []
    for r in scales:
        p = _block_reduce(gc, r, np.sum).ravel()
        p = p[p > 0]
        P.append(p / p.sum())

    alphas, fs = [], []
    for q in q_values:
        a_num, f_num = [], []
        for pr in P:
            denom = np.sum(pr ** q)
            if not np.isfinite(denom) or denom <= 0:
                a_num.append(np.nan); f_num.append(np.nan); continue
            mu = pr ** q / denom
            with np.errstate(divide="ignore", invalid="ignore"):
                a_num.append(np.sum(mu * np.log(pr)))
                f_num.append(np.sum(mu * np.log(mu)))
        a_num, f_num = np.array(a_num), np.array(f_num)
        ok = np.isfinite(a_num) & np.isfinite(f_num)
        if ok.sum() < 2:
            continue
        alphas.append(_slope(log_r[ok], a_num[ok]))
        fs.append(_slope(log_r[ok], f_num[ok]))
    width = float(max(alphas) - min(alphas)) if alphas else float("nan")
    return alphas, fs, width


# --------------------------------------------------------------------------------------
# Local FD map
# --------------------------------------------------------------------------------------
def local_fd_map(image: np.ndarray, window: int = 64, stride: int = 32) -> np.ndarray:
    g = np.asarray(image, dtype=np.float64)
    h, w = g.shape
    rows = max(1, 1 + (h - window) // stride)
    cols = max(1, 1 + (w - window) // stride)
    m = np.full((rows, cols), np.nan)
    for i in range(rows):
        for j in range(cols):
            patch = g[i * stride:i * stride + window, j * stride:j * stride + window]
            if patch.shape[0] < 8 or patch.shape[1] < 8:
                continue
            m[i, j], _ = dbc_fd(patch, min_box=2, max_box=window // 2)
    if np.isnan(m).any():
        m = np.where(np.isnan(m), np.nanmean(m), m)
    return m


# --------------------------------------------------------------------------------------
# Whole feature set
# --------------------------------------------------------------------------------------
FEATURE_NAMES = [
    "box_counting_fd", "dbc_fd", "lacunarity", "lacunarity_4", "lacunarity_8",
    "lacunarity_32", "lacunarity_64", "multifractal_width", "local_fd_mean", "local_fd_std",
]


def feature_dict(gray: np.ndarray, fd_map: np.ndarray | None = None) -> dict:
    """Same feature definitions as ``fractal.compute_fractal_features`` (+ the lacunarity curve)."""
    g = np.asarray(gray, dtype=np.float64)
    binary = g > g.mean()
    bc, _ = box_counting_fd(binary)
    db, _ = dbc_fd(g)
    lac = lacunarity_curve(binary)
    _, _, width = multifractal(g)
    if fd_map is None:
        fd_map = local_fd_map(g)
    lac_d = dict(lac)
    return {
        "box_counting_fd": bc,
        "dbc_fd": db,
        "lacunarity": lac_d[16],
        "lacunarity_4": lac_d[4],
        "lacunarity_8": lac_d[8],
        "lacunarity_32": lac_d[32],
        "lacunarity_64": lac_d[64],
        "multifractal_width": width,
        "local_fd_mean": float(np.nanmean(fd_map)),
        "local_fd_std": float(np.nanstd(fd_map)),
    }
