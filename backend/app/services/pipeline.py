"""
End-to-end analysis pipeline.

preprocess -> fractal features (+ local FD map) -> CNN (probs + embedding + activations)
-> fusion refine -> explainability overlays -> assembled AnalysisResponse.

Designed to degrade gracefully: if the ML stack is missing, it still returns the full
fractal analysis and overlays, with an empty prediction list and an explanatory note.
"""
from __future__ import annotations

import time
from typing import List

import numpy as np

from app.models.schemas import (
    AnalysisResponse,
    FractalMetrics,
    LacunarityPoint,
    Meta,
    Overlays,
    Prediction,
    SpectrumPoint,
    TopFinding,
)
from app.services import fractal as fx
from app.services.cnn_model import model as cnn
from app.services.fusion import head as fusion_head
from app.services.gradcam import activation_cam
from app.services.preprocessing import preprocess
from app.services.segmentation import bounding_box, segmenter
from app.services.validation import NotAnXrayError, assess, mean_saturation
from app.utils import images as im


def _severity(prob: float) -> str:
    if prob >= 0.6:
        return "high"
    if prob >= 0.35:
        return "moderate"
    return "low"


def analyze(data: bytes, filename: str) -> AnalysisResponse:
    t0 = time.time()

    pre = preprocess(data, filename)
    gray = pre.gray_full

    # --- Input validation, part 1: color -------------------------------------------
    # Reject obvious non-X-rays (color photos/screenshots) before doing any heavy work.
    saturation = mean_saturation(data, filename)
    early = assess(saturation, seg_available=False, lung_fraction=0.0)
    if not early.is_xray:
        raise NotAnXrayError(early.reason)

    # --- Lung segmentation (optional) -----------------------------------------------
    # Restrict fractal analysis to the lung fields when the segmentation model is present.
    lung_mask = None
    seg_applied = False
    lung_fraction = 0.0
    try:
        mask = segmenter.lung_mask(gray)
        if mask is not None:
            lung_fraction = float(mask.sum()) / mask.size
            if lung_fraction > 0.03:
                lung_mask = mask
                seg_applied = True
    except Exception:
        lung_mask = None

    # --- Input validation, part 2: lungs -------------------------------------------
    # When the segmenter is available, a real chest X-ray must contain detectable lungs.
    verdict = assess(saturation, seg_available=segmenter.available, lung_fraction=lung_fraction)
    if not verdict.is_xray:
        raise NotAnXrayError(verdict.reason)

    # --- Fractal branch -------------------------------------------------------------
    if seg_applied:
        bbox = bounding_box(lung_mask)
        r0, r1, c0, c1 = bbox
        roi = gray[r0:r1, c0:c1]
        feats = fx.compute_fractal_features(roi)  # lung-focused features
    else:
        feats = fx.compute_fractal_features(gray)
    fd_map = fx.local_fd_map(gray, window=64, stride=32)

    fractal_metrics = FractalMetrics(
        boxCountingFD=_finite(feats.box_counting_fd),
        dbcFD=_finite(feats.dbc_fd),
        lacunarity=_finite(feats.lacunarity),
        multifractalWidth=_finite(feats.multifractal_width),
        localFdMean=_finite(feats.local_fd_mean),
        localFdStd=_finite(feats.local_fd_std),
        spectrum=[SpectrumPoint(alpha=_finite(a), falpha=_finite(f)) for a, f in feats.spectrum],
        lacunarityCurve=[LacunarityPoint(boxSize=s, value=_finite(v)) for s, v in feats.lacunarity_curve],
    )

    # --- CNN branch + fusion --------------------------------------------------------
    predictions: List[Prediction] = []
    top: TopFinding | None = None
    gradcam_uri = None
    ml_available = cnn.available
    notes = None

    if ml_available:
        try:
            cnn_res = cnn.predict(pre.cnn_input)
            refined = fusion_head.refine(cnn_res.embedding, feats.to_vector(), cnn_res.probabilities)
            predictions = [
                Prediction(label=_pretty(k), probability=float(v))
                for k, v in sorted(refined.items(), key=lambda kv: kv[1], reverse=True)
            ]
            if predictions:
                p0 = predictions[0]
                top = TopFinding(label=p0.label, probability=p0.probability, severity=_severity(p0.probability))
            if cnn_res.activations is not None:
                cam = activation_cam(cnn_res.activations)
                gradcam_uri = im.encode_png_base64(im.overlay_heatmap(gray, cam, alpha=0.5))
            notes = "Fusion head applied." if fusion_head.available else "Pretrained multi-label CNN + fractal texture analysis."
        except Exception as e:
            ml_available = False
            notes = f"ML inference failed, showing fractal-only analysis: {e}"
    else:
        notes = "ML stack unavailable — fractal-only analysis. Install torch + torchxrayvision for predictions."

    # --- Overlays -------------------------------------------------------------------
    overlays = Overlays(
        original=im.encode_png_base64(pre.original_uint8),
        localFdMap=im.encode_png_base64(im.overlay_heatmap(gray, fd_map, alpha=0.6, mask=lung_mask)),
        gradcam=gradcam_uri,
        lungMask=im.encode_png_base64(im.lung_overlay(gray, lung_mask)) if seg_applied else None,
    )

    if seg_applied:
        seg_note = "Analysis focused on segmented lung fields."
        notes = f"{seg_note} {notes}" if notes else seg_note

    # --- Plain-language impression (how to read the numbers) ------------------------
    impression, significant = _impression(predictions, ml_available)

    meta = Meta(
        model=(cnn.name + (" + fusion" if fusion_head.available else "")) if ml_available else "fractal-only",
        processingMs=int((time.time() - t0) * 1000),
        mlAvailable=ml_available,
        segmentation=seg_applied,
        notes=notes,
    )

    return AnalysisResponse(
        predictions=predictions,
        topFinding=top,
        impression=impression,
        significant=significant,
        fractal=fractal_metrics,
        overlays=overlays,
        meta=meta,
    )


# The classifier has NO "Normal" output — it only scores diseases — and its scores are
# uncalibrated, so ~0.5 means "no real signal", not "moderately present". These thresholds
# turn the raw scores into an honest, readable impression.
STRONG_THRESHOLD = 0.65
BORDERLINE_THRESHOLD = 0.55


def _impression(predictions: List[Prediction], ml_available: bool) -> tuple[str, bool]:
    if not ml_available or not predictions:
        return ("Fractal texture analysis only (the disease model is not loaded).", False)
    top = predictions[0]
    pct = round(top.probability * 100)
    if top.probability >= STRONG_THRESHOLD:
        others = [p.label for p in predictions[1:3] if p.probability >= BORDERLINE_THRESHOLD]
        extra = (" Also possibly: " + ", ".join(others) + ".") if others else ""
        return (f"Likely abnormal — most probable finding is {top.label} ({pct}%).{extra}", True)
    if top.probability >= BORDERLINE_THRESHOLD:
        return (
            f"Borderline / uncertain — the model weakly suggests {top.label} ({pct}%) "
            f"but is not confident. Not clearly abnormal.",
            True,
        )
    return (
        "No clear abnormality detected — the lungs appear largely normal "
        "(all disease scores are low, which for this model means 'nothing obvious').",
        False,
    )


def _finite(x: float) -> float:
    return float(x) if np.isfinite(x) else 0.0


def _pretty(label: str) -> str:
    """TorchXRayVision labels are already human-readable (e.g. 'Effusion'); pass through."""
    return label
