"""
Fractal-dimension analysis for X-ray texture.

This module is the scientific heart of the project. It is deliberately dependency-light
(NumPy + SciPy only) so it can be unit-tested against mathematical fractals with known
dimensions (see backend/tests/test_fractal.py).

Implemented measures
--------------------
- box_counting_dimension        : classic FD of a binary set (Minkowski-Bouligand).
- differential_box_counting     : DBC (Sarkar & Chaudhuri) FD of a *grayscale* surface.
- lacunarity                    : gliding-box lacunarity Lambda(r) = E[M^2] / E[M]^2.
- multifractal_spectrum         : the f(alpha) spectrum via the method of moments.
- local_fd_map                  : per-window DBC over the image -> 2-D FD map (the star).

All functions expect 2-D NumPy arrays. Grayscale intensities may be uint8 [0,255] or
float; they are handled internally.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _fit_slope(log_x: np.ndarray, log_y: np.ndarray) -> float:
    """Least-squares slope of log_y vs log_x. Returns the fractal dimension estimate."""
    if len(log_x) < 2:
        return float("nan")
    slope, _ = np.polyfit(log_x, log_y, 1)
    return float(slope)


def _power_of_two_sizes(min_size: int, max_size: int) -> List[int]:
    """Box sizes as powers of two within [min_size, max_size]."""
    sizes: List[int] = []
    s = min_size
    while s <= max_size:
        sizes.append(s)
        s *= 2
    return sizes


def _to_gray_float(image: np.ndarray) -> np.ndarray:
    """Collapse to 2-D float. Accepts (H,W) or (H,W,C)."""
    arr = np.asarray(image)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    return arr.astype(np.float64)


# --------------------------------------------------------------------------------------
# 1. Box-counting dimension (binary set)
# --------------------------------------------------------------------------------------
def box_counting_dimension(
    binary: np.ndarray,
    min_box: int = 2,
    max_box: int | None = None,
) -> Tuple[float, List[Tuple[int, int]]]:
    """
    Minkowski-Bouligand (box-counting) dimension of a binary image.

    Uses the canonical vectorised reduction: for each box side `s`, sum occupancy over
    non-overlapping s x s blocks with ``np.add.reduceat`` and count the non-empty blocks.
    The dimension is ``-slope`` of ``log(count)`` vs ``log(size)`` (since N(s) ~ s^-D).

    Parameters
    ----------
    binary   : 2-D array; non-zero pixels are treated as "occupied".
    min_box  : smallest box side (pixels).
    max_box  : largest box side; defaults to the largest power of two <= min(H, W)//2.

    Returns
    -------
    (fd, curve) where curve is a list of (box_size, box_count) pairs used for the fit,
    handy for plotting the log-log line in the UI / report.
    """
    b = (np.asarray(binary) != 0)
    h, w = b.shape
    if max_box is None:
        max_box = max(min_box, 2 ** int(np.floor(np.log2(min(h, w) / 2 or 1))))

    sizes = _power_of_two_sizes(min_box, max_box)
    curve: List[Tuple[int, int]] = []
    for size in sizes:
        # Sum occupancy over non-overlapping size x size blocks, then count non-empty ones.
        reduced = np.add.reduceat(
            np.add.reduceat(b, np.arange(0, h, size), axis=0),
            np.arange(0, w, size), axis=1,
        )
        count = int(np.count_nonzero(reduced))
        curve.append((size, max(count, 1)))

    log_size = np.array([np.log(s) for s, _ in curve])
    log_counts = np.array([np.log(n) for _, n in curve])
    fd = -_fit_slope(log_size, log_counts)
    return fd, curve


# --------------------------------------------------------------------------------------
# 2. Differential Box Counting (grayscale surface) -- Sarkar & Chaudhuri
# --------------------------------------------------------------------------------------
def differential_box_counting(
    image: np.ndarray,
    min_box: int = 2,
    max_box: int | None = None,
) -> Tuple[float, List[Tuple[int, int]]]:
    """
    DBC fractal dimension of a grayscale intensity surface.

    The image is treated as a 3-D surface (x, y, intensity). For each grid box of side r,
    intensities are quantised into vertical columns of height r' = r * (G / W), and the
    contribution n_r for the box is (highest column - lowest column occupied) + 1. The FD
    is the slope of log(sum n_r) vs log(1/r).

    Returns (fd, curve) with curve = [(box_size, summed_count), ...].
    """
    g = _to_gray_float(image)
    # Normalise intensities to [0, 255] grey levels G.
    gmin, gmax = float(g.min()), float(g.max())
    if gmax - gmin < 1e-9:
        return float("nan"), []
    G = 255.0
    g = (g - gmin) / (gmax - gmin) * G

    h, w = g.shape
    W = min(h, w)
    if max_box is None:
        max_box = max(min_box, 2 ** int(np.floor(np.log2(W / 2 or 1))))

    sizes = _power_of_two_sizes(min_box, max_box)
    curve: List[Tuple[int, int]] = []
    for r in sizes:
        # Vertical box height in grey levels for this scale.
        h_r = max(1.0, G * r / W)
        nrows = int(np.ceil(h / r))
        ncols = int(np.ceil(w / r))
        total = 0
        for i in range(nrows):
            r0, r1 = i * r, min((i + 1) * r, h)
            for j in range(ncols):
                c0, c1 = j * r, min((j + 1) * r, w)
                block = g[r0:r1, c0:c1]
                # Column index (level) of the min and max intensity in this box.
                k_min = np.floor(block.min() / h_r)
                k_max = np.floor(block.max() / h_r)
                total += int(k_max - k_min + 1)
        curve.append((r, max(total, 1)))

    log_inv_size = np.array([np.log(1.0 / s) for s, _ in curve])
    log_counts = np.array([np.log(n) for _, n in curve])
    fd = _fit_slope(log_inv_size, log_counts)
    return fd, curve


# --------------------------------------------------------------------------------------
# 3. Lacunarity (gliding-box)
# --------------------------------------------------------------------------------------
def lacunarity(binary: np.ndarray, box_size: int = 16) -> float:
    """
    Gliding-box lacunarity for a single box size.

    Lambda(r) = E[M^2] / (E[M])^2, where M is the mass (occupied-pixel count) in a box of
    side r as it glides over the image. Lambda == 1 for a perfectly homogeneous set;
    higher values mean more "gappy"/heterogeneous texture. Complements FD.
    """
    b = (np.asarray(binary) != 0).astype(np.float64)
    h, w = b.shape
    if box_size >= min(h, w):
        box_size = max(1, min(h, w) // 2)

    # Integral image for O(1) box sums.
    ii = _integral_image(b)
    masses: List[float] = []
    for i in range(0, h - box_size + 1):
        for j in range(0, w - box_size + 1):
            masses.append(_box_sum(ii, i, j, box_size, box_size))
    m = np.asarray(masses)
    mean = m.mean()
    if mean <= 0:
        return float("nan")
    return float((m ** 2).mean() / (mean ** 2))


def lacunarity_curve(binary: np.ndarray, box_sizes: List[int] | None = None) -> List[Tuple[int, float]]:
    """Lacunarity across several box sizes -- a texture 'fingerprint' for the report."""
    if box_sizes is None:
        box_sizes = [4, 8, 16, 32, 64]
    out: List[Tuple[int, float]] = []
    for r in box_sizes:
        out.append((r, lacunarity(binary, r)))
    return out


def _integral_image(a: np.ndarray) -> np.ndarray:
    return np.pad(np.cumsum(np.cumsum(a, axis=0), axis=1), ((1, 0), (1, 0)))


def _box_sum(ii: np.ndarray, r: int, c: int, hh: int, ww: int) -> float:
    return float(ii[r + hh, c + ww] - ii[r, c + ww] - ii[r + hh, c] + ii[r, c])


# --------------------------------------------------------------------------------------
# 4. Multifractal spectrum f(alpha) -- method of moments
# --------------------------------------------------------------------------------------
@dataclass
class MultifractalResult:
    alpha: List[float] = field(default_factory=list)     # singularity strength
    f_alpha: List[float] = field(default_factory=list)   # spectrum
    q: List[float] = field(default_factory=list)
    width: float = float("nan")                          # alpha_max - alpha_min (Delta-alpha)


def multifractal_spectrum(
    image: np.ndarray,
    q_values: np.ndarray | None = None,
    min_box: int = 2,
    max_box: int | None = None,
) -> MultifractalResult:
    """
    Estimate the multifractal spectrum f(alpha) of a grayscale image via the method of
    moments (Chhabra & Jensen style, direct estimation).

    For each scale r and moment q we form normalised measures mu_i(q, r) from box masses,
    then estimate alpha(q) and f(q) as slopes vs log(r). The spectrum *width* (Delta-alpha)
    is a compact, discriminative feature: broader spectra indicate richer, more
    heterogeneous texture.
    """
    g = _to_gray_float(image)
    g = g - g.min()
    if g.max() <= 0:
        return MultifractalResult()
    if q_values is None:
        q_values = np.linspace(-5, 5, 21)

    h, w = g.shape
    W = min(h, w)
    if max_box is None:
        max_box = max(min_box, 2 ** int(np.floor(np.log2(W / 2 or 1))))
    scales = _power_of_two_sizes(min_box, max_box)
    if len(scales) < 2:
        return MultifractalResult()

    # Pre-compute box masses P_i(r) at every scale.
    masses_by_scale: dict[int, np.ndarray] = {}
    for r in scales:
        ii = _integral_image(g)
        vals = []
        for i in range(0, h - r + 1, r):
            for j in range(0, w - r + 1, r):
                vals.append(_box_sum(ii, i, j, r, r))
        p = np.asarray(vals, dtype=np.float64)
        p = p[p > 0]
        masses_by_scale[r] = p

    log_r = np.array([np.log(r) for r in scales])
    alphas, fs, qs = [], [], []
    for q in q_values:
        alpha_num, f_num = [], []
        for r in scales:
            p = masses_by_scale[r]
            total = p.sum()
            if total <= 0 or len(p) == 0:
                alpha_num.append(np.nan)
                f_num.append(np.nan)
                continue
            pr = p / total
            # normalised measure mu_i(q)
            denom = np.sum(pr ** q)
            if denom <= 0:
                alpha_num.append(np.nan)
                f_num.append(np.nan)
                continue
            mu = (pr ** q) / denom
            with np.errstate(divide="ignore", invalid="ignore"):
                alpha_num.append(np.sum(mu * np.log(pr)))
                f_num.append(np.sum(mu * np.log(mu)))
        alpha_num = np.asarray(alpha_num)
        f_num = np.asarray(f_num)
        valid = np.isfinite(alpha_num) & np.isfinite(f_num)
        if valid.sum() < 2:
            continue
        a = _fit_slope(log_r[valid], alpha_num[valid])
        f = _fit_slope(log_r[valid], f_num[valid])
        alphas.append(a)
        fs.append(f)
        qs.append(float(q))

    width = float(max(alphas) - min(alphas)) if alphas else float("nan")
    return MultifractalResult(alpha=alphas, f_alpha=fs, q=qs, width=width)


# --------------------------------------------------------------------------------------
# 5. Local FD map (the signature visualization)
# --------------------------------------------------------------------------------------
def local_fd_map(
    image: np.ndarray,
    window: int = 64,
    stride: int = 32,
) -> np.ndarray:
    """
    Compute a per-window DBC fractal dimension over the image, producing a coarse 2-D map
    of local texture complexity. Upsample + colormap this in the UI to *show where* the
    texture is abnormal -- the project's signature explainability visual.

    Returns a float 2-D array (rows x cols of windows) of local FD values.
    """
    g = _to_gray_float(image)
    h, w = g.shape
    rows = max(1, 1 + (h - window) // stride)
    cols = max(1, 1 + (w - window) // stride)
    fd_map = np.zeros((rows, cols), dtype=np.float64)
    for i in range(rows):
        for j in range(cols):
            r0 = i * stride
            c0 = j * stride
            patch = g[r0:r0 + window, c0:c0 + window]
            if patch.shape[0] < 4 or patch.shape[1] < 4:
                fd_map[i, j] = np.nan
                continue
            fd, _ = differential_box_counting(patch, min_box=2, max_box=window // 2)
            fd_map[i, j] = fd
    # Replace NaNs with the map mean for a clean visualization.
    if np.isnan(fd_map).any():
        mean = np.nanmean(fd_map)
        fd_map = np.where(np.isnan(fd_map), mean, fd_map)
    return fd_map


# --------------------------------------------------------------------------------------
# Aggregate feature vector (fed to the fusion head)
# --------------------------------------------------------------------------------------
@dataclass
class FractalFeatures:
    box_counting_fd: float
    dbc_fd: float
    lacunarity: float
    lacunarity_curve: List[Tuple[int, float]]
    multifractal_width: float
    spectrum: List[Tuple[float, float]]   # (alpha, f_alpha) points
    local_fd_mean: float
    local_fd_std: float

    def to_vector(self) -> np.ndarray:
        """Compact numeric vector for the fusion MLP (order is stable -- do not reorder)."""
        lac_vals = [v for _, v in self.lacunarity_curve]
        base = [
            self.box_counting_fd,
            self.dbc_fd,
            self.lacunarity,
            self.multifractal_width,
            self.local_fd_mean,
            self.local_fd_std,
        ]
        return np.nan_to_num(np.asarray(base + lac_vals, dtype=np.float32))


def compute_fractal_features(gray: np.ndarray, binary: np.ndarray | None = None) -> FractalFeatures:
    """
    Compute the full fractal feature set for one preprocessed grayscale X-ray.

    `binary` is an optional occupancy mask (e.g. edge map or Otsu threshold). If omitted a
    simple mean-threshold binarization is used for the box-counting / lacunarity measures.
    """
    g = _to_gray_float(gray)
    if binary is None:
        binary = g > g.mean()

    bc_fd, _ = box_counting_dimension(binary)
    dbc_fd, _ = differential_box_counting(g)
    lac = lacunarity(binary, box_size=16)
    lac_curve = lacunarity_curve(binary)
    mf = multifractal_spectrum(g)
    fd_map = local_fd_map(g)

    return FractalFeatures(
        box_counting_fd=bc_fd,
        dbc_fd=dbc_fd,
        lacunarity=lac,
        lacunarity_curve=lac_curve,
        multifractal_width=mf.width,
        spectrum=list(zip(mf.alpha, mf.f_alpha)),
        local_fd_mean=float(np.nanmean(fd_map)),
        local_fd_std=float(np.nanstd(fd_map)),
    )
