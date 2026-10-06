"""
The fast (vectorised) fractal code must agree with the readable reference implementation,
and must still recover the known dimensions of mathematical fractals.

Run:  cd backend && pytest -q tests/test_fractal_fast.py
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services import fractal as ref  # noqa: E402
from app.services import fractal_fast as fast  # noqa: E402
from tests import fractal_generators as gen  # noqa: E402


@pytest.fixture(scope="module")
def texture():
    rng = np.random.default_rng(0)
    base = rng.random((130, 130))
    # smooth a little so the texture is not pure white noise
    k = np.ones((3, 3)) / 9
    from scipy.signal import convolve2d
    return convolve2d(base, k, mode="valid")[:128, :128]


def test_box_counting_matches_reference(texture):
    b = texture > texture.mean()
    assert fast.box_counting_fd(b)[0] == pytest.approx(ref.box_counting_dimension(b)[0], abs=1e-9)


def test_dbc_matches_reference(texture):
    assert fast.dbc_fd(texture)[0] == pytest.approx(ref.differential_box_counting(texture)[0], abs=1e-9)


def test_lacunarity_matches_reference(texture):
    b = texture > texture.mean()
    assert fast.lacunarity(b, 8) == pytest.approx(ref.lacunarity(b, 8), rel=1e-9)


def test_multifractal_width_matches_reference(texture):
    assert fast.multifractal(texture)[2] == pytest.approx(ref.multifractal_spectrum(texture).width, rel=1e-6)


def test_local_fd_map_matches_reference(texture):
    a = fast.local_fd_map(texture, 64, 32)
    b = ref.local_fd_map(texture, 64, 32)
    assert np.allclose(a, b, atol=1e-9)


def test_fast_sierpinski_triangle():
    img = gen.sierpinski_triangle(order=8)
    assert fast.box_counting_fd(img)[0] == pytest.approx(math.log(3) / math.log(2), abs=0.08)


def test_fast_koch_curve():
    img = gen.koch_curve(order=4, size=2048)
    assert fast.box_counting_fd(img)[0] == pytest.approx(math.log(4) / math.log(3), abs=0.06)
