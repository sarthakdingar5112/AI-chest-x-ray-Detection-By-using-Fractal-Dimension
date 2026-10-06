"""
Lung-field segmentation via TorchXRayVision's pretrained PSPNet (chestx_det).

Produces a boolean lung mask so the fractal analysis can be restricted to the lung
parenchyma — where pathology-relevant texture lives — instead of ribs, diaphragm and
borders. This makes the FD / lacunarity / multifractal features cleaner and more clinically
meaningful, and yields a heatmap that only lights up inside the lungs.

Torch imports are lazy; if the model or torch is unavailable, `lung_mask` returns None and
the pipeline falls back to whole-image analysis.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

try:
    from PIL import Image
    _PIL = True
except Exception:
    _PIL = False


class LungSegmenter:
    def __init__(self):
        self._model = None
        self._torch = None
        self._xrv = None
        self._targets: List[str] = []
        self._error: Optional[str] = None

    @property
    def available(self) -> bool:
        if self._model is None and self._error is None:
            self._try_load()
        return self._model is not None

    def _try_load(self) -> None:
        try:
            from app.core.console import ensure_utf8_output
            ensure_utf8_output()  # prevent Windows cp1252 crash during weight download
            import torch
            import torchxrayvision as xrv
            self._torch = torch
            self._xrv = xrv
            model = xrv.baseline_models.chestx_det.PSPNet()
            model.eval()
            self._model = model
            self._targets = list(model.targets)
        except Exception as e:  # torch missing, no weights cache, etc.
            self._error = str(e)

    def lung_mask(self, gray_full: np.ndarray, threshold: float = 0.5) -> Optional[np.ndarray]:
        """
        Parameters
        ----------
        gray_full : float grayscale image in [0, 1], display resolution.

        Returns a boolean lung mask at `gray_full` resolution (union of left + right lung),
        or None if segmentation is unavailable.
        """
        if not self.available or not _PIL:
            return None
        torch, xrv = self._torch, self._xrv
        h, w = gray_full.shape

        # PSPNet expects a single-channel 512x512 image normalized to [-1024, 1024].
        img8 = (np.clip(gray_full, 0, 1) * 255).astype(np.float32)
        resized = np.asarray(Image.fromarray(img8.astype(np.uint8)).resize((512, 512), Image.BILINEAR), dtype=np.float32)
        norm = xrv.datasets.normalize(resized, 255)  # -> [-1024, 1024]
        x = torch.from_numpy(norm)[None, None, :, :]

        with torch.no_grad():
            out = self._model(x)
            pred = torch.sigmoid(out)[0].cpu().numpy()  # (n_targets, 512, 512)

        lung_idx = [i for i, t in enumerate(self._targets) if "Lung" in t]
        if not lung_idx:
            return None
        lung = np.zeros((512, 512), dtype=np.float32)
        for i in lung_idx:
            lung = np.maximum(lung, pred[i])
        mask512 = (lung > threshold).astype(np.uint8) * 255

        # Resize mask back to display resolution (nearest to keep it binary).
        mask_full = np.asarray(Image.fromarray(mask512).resize((w, h), Image.NEAREST)) > 127
        return mask_full


def bounding_box(mask: np.ndarray, pad: int = 4) -> Optional[tuple[int, int, int, int]]:
    """Return (r0, r1, c0, c1) bounding box of a boolean mask, padded, or None if empty."""
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    r0, r1 = max(0, ys.min() - pad), min(mask.shape[0], ys.max() + pad + 1)
    c0, c1 = max(0, xs.min() - pad), min(mask.shape[1], xs.max() + pad + 1)
    return int(r0), int(r1), int(c0), int(c1)


# Module-level singleton.
segmenter = LungSegmenter()
