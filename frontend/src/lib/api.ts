import type { AnalysisResponse } from "../types/analysis";

const BASE = import.meta.env.VITE_API_BASE ?? "";

export async function analyzeImage(file: File): Promise<AnalysisResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/api/analyze`, { method: "POST", body: form });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = await res.json();
      detail = j.detail ?? detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json();
}

export async function health(): Promise<{ status: string; mlAvailable: boolean; model: string | null }> {
  const res = await fetch(`${BASE}/api/health`);
  return res.json();
}
