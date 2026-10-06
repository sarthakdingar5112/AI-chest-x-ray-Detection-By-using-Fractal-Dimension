// Mirrors backend/app/models/schemas.py

export interface Prediction {
  label: string;
  probability: number;
}

export interface TopFinding {
  label: string;
  probability: number;
  severity: "low" | "moderate" | "high";
}

export interface SpectrumPoint {
  alpha: number;
  falpha: number;
}

export interface LacunarityPoint {
  boxSize: number;
  value: number;
}

export interface FractalMetrics {
  boxCountingFD: number;
  dbcFD: number;
  lacunarity: number;
  multifractalWidth: number;
  localFdMean: number;
  localFdStd: number;
  spectrum: SpectrumPoint[];
  lacunarityCurve: LacunarityPoint[];
}

export interface Overlays {
  original: string;
  localFdMap: string;
  gradcam: string | null;
  lungMask: string | null;
}

export interface Meta {
  model: string;
  processingMs: number;
  mlAvailable: boolean;
  segmentation: boolean;
  notes?: string | null;
}

export interface AnalysisResponse {
  predictions: Prediction[];
  topFinding: TopFinding | null;
  impression: string;
  significant: boolean;
  fractal: FractalMetrics;
  overlays: Overlays;
  meta: Meta;
}
