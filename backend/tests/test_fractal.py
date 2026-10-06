"""
Validation of the fractal-dimension implementation against fractals with KNOWN dimension.

Run:  cd backend && pytest -q

These tests are the correctness proof for the whole project. The box-counting estimate is
approximate (finite resolution, discrete scales), so tolerances are generous but the
estimates must land clearly nearer the true value than to a neighbouring integer.
"""
import math
import os
import sys

import numpy as np
import pytest

# Make `app` importable when running pytest from the backend/ directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services import fractal  # noqa: E402
from tests import fractal_generators as gen  # noqa: E402


SIERPINSKI_TRIANGLE_D = math.log(3) / math.log(2)   # ~1.585
SIERPINSKI_CARPET_D = math.log(8) / math.log(3)      # ~1.893
KOCH_D = math.log(4) / math.log(3)                   # ~1.262


def test_sierpinski_triangle_dimension():
    img = gen.sierpinski_triangle(order=8)
    fd, curve = fractal.box_counting_dimension(img, min_box=2)
    assert len(curve) >= 4
    assert fd == pytest.approx(SIERPINSKI_TRIANGLE_D, abs=0.08)


def test_sierpinski_carpet_dimension():
    # order 5 (243x243) is well resolved for power-of-two box counting.
    img = gen.sierpinski_carpet(order=5)
    fd, _ = fractal.box_counting_dimension(img, min_box=2)
    assert fd == pytest.approx(SIERPINSKI_CARPET_D, abs=0.06)


def test_koch_curve_dimension():
    # order 4 on a 2048px canvas keeps every segment several pixels wide, so the fractal
    # structure is resolved across the box-size range (sub-pixel segments inflate the FD).
    img = gen.koch_curve(order=4, size=2048)
    fd, _ = fractal.box_counting_dimension(img, min_box=2)
    assert fd == pytest.approx(KOCH_D, abs=0.06)


def test_filled_square_is_two_dimensional():
    """A solid filled square should have box-counting dimension ~2."""
    img = np.ones((256, 256), dtype=np.uint8)
    fd, _ = fractal.box_counting_dimension(img, min_box=2)
    assert fd == pytest.approx(2.0, abs=0.05)


def test_single_line_is_one_dimensional():
    """A straight line should have dimension ~1."""
    img = np.zeros((256, 256), dtype=np.uint8)
    img[128, :] = 1
    fd, _ = fractal.box_counting_dimension(img, min_box=2)
    assert fd == pytest.approx(1.0, abs=0.10)


def test_lacunarity_homogeneous_lower_than_clustered():
    """Uniform texture is less lacunar (==1) than a genuinely gappy/blocky one."""
    uniform = np.ones((128, 128), dtype=np.uint8)
    # A block checkerboard has real variance in box mass (some boxes full, some empty),
    # so its gliding-box lacunarity must exceed 1.
    clustered = np.zeros((128, 128), dtype=np.uint8)
    block = 32
    for i in range(0, 128, block):
        for j in range(0, 128, block):
            if ((i // block) + (j // block)) % 2 == 0:
                clustered[i:i + block, j:j + block] = 1
    lac_uniform = fractal.lacunarity(uniform, box_size=16)
    lac_clustered = fractal.lacunarity(clustered, box_size=16)
    assert lac_uniform == pytest.approx(1.0, abs=0.05)
    assert lac_clustered > lac_uniform


def test_multifractal_width_nonnegative():
    rng = np.random.default_rng(0)
    img = rng.random((128, 128))
    mf = fractal.multifractal_spectrum(img)
    assert mf.width >= 0 or math.isnan(mf.width)


def test_local_fd_map_shape_and_range():
    rng = np.random.default_rng(1)
    img = (rng.random((256, 256)) * 255).astype(np.uint8)
    fmap = fractal.local_fd_map(img, window=64, stride=32)
    assert fmap.ndim == 2
    assert fmap.shape[0] >= 1 and fmap.shape[1] >= 1
    assert np.isfinite(fmap).all()


def test_feature_vector_is_finite_and_stable_length():
    rng = np.random.default_rng(2)
    img = (rng.random((256, 256)) * 255).astype(np.uint8)
    feats = fractal.compute_fractal_features(img)
    vec = feats.to_vector()
    assert np.isfinite(vec).all()
    # 6 scalar features + 5 lacunarity-curve entries = 11.
    assert vec.shape[0] == 11
