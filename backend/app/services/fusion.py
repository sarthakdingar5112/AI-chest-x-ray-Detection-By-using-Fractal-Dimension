"""
Fusion head: refine CNN predictions using the fractal feature vector.

At training time (notebooks/02_train_fusion.ipynb) a small MLP is trained on
[CNN embedding (1024) || fractal vector (~11)] -> per-pathology logits, and exported to
`backend/weights/fusion_head.pt`.

At serving time, if the weights and torch are present we run the MLP; otherwise we fall
back to the raw CNN probabilities (graceful degradation) so the app always works.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "weights", "fusion_head.pt")


class FusionHead:
    def __init__(self, weights_path: str = WEIGHTS_PATH):
        self.weights_path = os.path.abspath(weights_path)
        self._model = None
        self._torch = None
        self._labels: List[str] = []
        self._tried = False

    @property
    def available(self) -> bool:
        if not self._tried:
            self._try_load()
        return self._model is not None

    def _try_load(self) -> None:
        self._tried = True
        if not os.path.exists(self.weights_path):
            return
        try:
            import torch
            self._torch = torch
            ckpt = torch.load(self.weights_path, map_location="cpu")
            self._labels = ckpt.get("labels", [])
            in_dim = ckpt["in_dim"]
            hidden = ckpt.get("hidden", 256)
            out_dim = len(self._labels)
            model = _build_mlp(torch, in_dim, hidden, out_dim)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            self._model = model
        except Exception:
            self._model = None

    def refine(
        self,
        embedding: np.ndarray,
        fractal_vector: np.ndarray,
        cnn_probs: Dict[str, float],
    ) -> Dict[str, float]:
        """Return refined probabilities; falls back to cnn_probs if head unavailable."""
        if not self.available:
            return cnn_probs
        torch = self._torch
        x = np.concatenate([embedding.astype("float32"), fractal_vector.astype("float32")])
        with torch.no_grad():
            logits = self._model(torch.from_numpy(x)[None, :])
            probs = torch.sigmoid(logits)[0].cpu().numpy()
        fused = {label: float(p) for label, p in zip(self._labels, probs)}
        # The head may be trained on only SOME labels (e.g. just "Pneumonia"). It overrides
        # those, and every other pathology keeps its pretrained-CNN probability.
        return {**cnn_probs, **fused}


def _build_mlp(torch, in_dim: int, hidden: int, out_dim: int):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden, out_dim),
    )


head = FusionHead()
