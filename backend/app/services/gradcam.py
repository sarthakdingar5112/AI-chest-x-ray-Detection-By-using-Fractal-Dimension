"""
Grad-CAM style attention map from the CNN's last convolutional features.

For a lightweight, dependency-safe implementation we use the activation magnitude of the
final DenseNet feature block (a "Grad-CAM lite" / activation map). This highlights the
image regions the network responds to most strongly and pairs with the fractal texture map
for the dual-explainability view. Given weighting logits, it computes a proper
class-weighted CAM.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def activation_cam(activations: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Parameters
    ----------
    activations : (C, H, W) last-conv feature maps.
    weights     : optional (C,) per-channel weights (e.g. from the top class). If None, a
                  simple mean over channels (activation energy) is used.

    Returns a normalized 2-D CAM in [0, 1].
    """
    if activations is None or activations.ndim != 3:
        raise ValueError("activations must be (C, H, W)")
    if weights is None:
        cam = np.maximum(activations, 0).mean(axis=0)
    else:
        w = weights.reshape(-1, 1, 1)
        cam = np.maximum((activations * w).sum(axis=0), 0)
    lo, hi = float(cam.min()), float(cam.max())
    if hi - lo < 1e-9:
        return np.zeros_like(cam)
    return (cam - lo) / (hi - lo)
