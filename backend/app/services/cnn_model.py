"""
CNN branch: pretrained multi-label chest X-ray classifier via TorchXRayVision.

Loads a DenseNet-121 (`densenet121-res224-all`) trained across NIH/CheXpert/MIMIC/PadChest.
Returns per-pathology probabilities plus the 1024-d feature embedding used by the fusion
head. All torch imports are lazy so the rest of the backend runs without ML installed.

If torch / torchxrayvision are unavailable the module reports `available == False` and the
pipeline falls back to fractal-only analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class CnnResult:
    probabilities: Dict[str, float]      # pathology -> prob
    embedding: np.ndarray                # 1024-d feature vector
    activations: Optional[np.ndarray]    # last conv feature maps (for Grad-CAM)


class ChestXRayModel:
    """Lazily-loaded singleton wrapper around a TorchXRayVision DenseNet."""

    def __init__(self, weights: str = "densenet121-res224-all"):
        self.weights = weights
        self._model = None
        self._xrv = None
        self._torch = None
        self._pathologies: List[str] = []
        self._load_error: Optional[str] = None

    @property
    def available(self) -> bool:
        if self._model is None and self._load_error is None:
            self._try_load()
        return self._model is not None

    @property
    def pathologies(self) -> List[str]:
        return self._pathologies

    @property
    def name(self) -> str:
        return self.weights

    def _try_load(self) -> None:
        try:
            from app.core.console import ensure_utf8_output
            ensure_utf8_output()  # prevent Windows cp1252 crash during weight download
            import torch
            import torchxrayvision as xrv
            self._torch = torch
            self._xrv = xrv
            model = xrv.models.DenseNet(weights=self.weights)
            model.eval()
            self._model = model
            self._pathologies = list(model.pathologies)
        except Exception as e:  # torch missing, no weights cache, etc.
            self._load_error = str(e)

    def predict(self, cnn_input: np.ndarray) -> CnnResult:
        """
        cnn_input : 224x224 float array already normalized to [-1024, 1024].
        """
        if not self.available:
            raise RuntimeError(f"CNN unavailable: {self._load_error}")
        torch = self._torch
        x = torch.from_numpy(cnn_input.astype("float32"))[None, None, :, :]  # (1,1,224,224)

        activations = {}

        def hook(_module, _inp, out):
            activations["feat"] = out.detach()

        # DenseNet-121 features live in model.features; hook the final block.
        handle = self._model.features.register_forward_hook(hook)
        try:
            with torch.no_grad():
                logits = self._model(x)
        finally:
            handle.remove()

        probs = logits[0].detach().cpu().numpy()
        prob_map = {
            p: float(v)
            for p, v in zip(self._pathologies, probs)
            if p  # skip empty label slots
        }

        feat = activations.get("feat")
        embedding = feat.mean(dim=[2, 3])[0].cpu().numpy() if feat is not None else np.zeros(1024, dtype=np.float32)
        acts = feat[0].cpu().numpy() if feat is not None else None
        return CnnResult(probabilities=prob_map, embedding=embedding, activations=acts)


# Module-level singleton, warm-loaded at app startup.
model = ChestXRayModel()
