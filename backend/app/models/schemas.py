"""Pydantic request/response schemas. Mirror these in frontend/src/types/analysis.ts."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class Prediction(BaseModel):
    label: str
    probability: float = Field(..., ge=0.0, le=1.0)


class TopFinding(BaseModel):
    label: str
    probability: float
    severity: str  # "low" | "moderate" | "high"


class SpectrumPoint(BaseModel):
    alpha: float
    falpha: float


class LacunarityPoint(BaseModel):
    boxSize: int
    value: float


class FractalMetrics(BaseModel):
    boxCountingFD: float
    dbcFD: float
    lacunarity: float
    multifractalWidth: float
    localFdMean: float
    localFdStd: float
    spectrum: List[SpectrumPoint] = []
    lacunarityCurve: List[LacunarityPoint] = []


class Overlays(BaseModel):
    original: str          # data:image/png;base64,...
    localFdMap: str
    gradcam: Optional[str] = None
    lungMask: Optional[str] = None


class Meta(BaseModel):
    model: str
    processingMs: int
    mlAvailable: bool
    segmentation: bool = False
    notes: Optional[str] = None


class AnalysisResponse(BaseModel):
    predictions: List[Prediction]
    topFinding: Optional[TopFinding] = None
    impression: str = ""          # plain-language summary of the result
    significant: bool = False     # True only when a finding is clearly above chance
    fractal: FractalMetrics
    overlays: Overlays
    meta: Meta


class HealthResponse(BaseModel):
    status: str
    mlAvailable: bool
    model: Optional[str] = None
