import { useState } from "react";
import { motion } from "framer-motion";
import type { AnalysisResponse } from "../types/analysis";
import HeatmapViewer from "./HeatmapViewer";
import PredictionList from "./PredictionList";
import ConfidenceGauge from "./ConfidenceGauge";
import FractalMetricsPanel from "./FractalMetricsPanel";
import MultifractalChart from "./MultifractalChart";
import { generateReport } from "../lib/report";

const Section = ({ title, children, delay = 0 }: { title: string; children: React.ReactNode; delay?: number }) => (
  <motion.section
    className="card"
    initial={{ opacity: 0, y: 16 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ delay, duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
    style={{ padding: 22 }}
  >
    <h3 style={{ fontSize: 15, color: "var(--text-1)", marginBottom: 16, letterSpacing: "0.02em" }}>{title}</h3>
    {children}
  </motion.section>
);

export default function AnalysisView({ result, onReset }: { result: AnalysisResponse; onReset: () => void }) {
  const [pdfBusy, setPdfBusy] = useState(false);

  const downloadPdf = async () => {
    setPdfBusy(true);
    try {
      await generateReport(result);
    } finally {
      setPdfBusy(false);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 10 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
          <span className="badge mono">{result.meta.model}</span>
          <span className="badge mono">{result.meta.processingMs} ms</span>
          {result.meta.segmentation && <span className="badge sev-low">lung-focused</span>}
          {!result.meta.mlAvailable && <span className="badge sev-moderate">fractal-only</span>}
        </div>
        <div style={{ display: "flex", gap: 10 }}>
          <button className="btn" onClick={downloadPdf} disabled={pdfBusy} style={{ background: "transparent" }}>
            {pdfBusy ? "Preparing…" : "⤓ Download PDF report"}
          </button>
          <button className="btn" onClick={onReset}>Analyze another</button>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "minmax(320px, 1.15fr) 1fr", gap: 18, alignItems: "start" }}>
        <Section title="Image & explainability overlays">
          <HeatmapViewer overlays={result.overlays} />
        </Section>

        <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
          <Section title="Diagnosis" delay={0.05}>
            {result.impression && <ImpressionBanner result={result} />}
            <div style={{ display: "flex", gap: 24, alignItems: "center", flexWrap: "wrap", marginTop: result.impression ? 16 : 0 }}>
              {result.significant && result.topFinding && <ConfidenceGauge finding={result.topFinding} />}
              <div style={{ flex: 1, minWidth: 220 }}>
                <PredictionList predictions={result.predictions} />
              </div>
            </div>
            {result.predictions.length > 0 && (
              <p className="muted" style={{ fontSize: 11.5, lineHeight: 1.6, marginTop: 14, borderTop: "1px solid var(--border)", paddingTop: 12 }}>
                <strong>How to read this:</strong> this AI has no "normal" option — it only scores diseases,
                so it always lists some. A score near <strong>50% means the model is unsure</strong> (not "half sick");
                only clearly high scores suggest a real finding.
              </p>
            )}
          </Section>
        </div>
      </div>

      <Section title="Fractal signature" delay={0.1}>
        <FractalMetricsPanel fractal={result.fractal} />
      </Section>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 18 }}>
        <Section title="Multifractal spectrum f(α)" delay={0.15}>
          <MultifractalChart spectrum={result.fractal.spectrum} />
          <p className="muted" style={{ fontSize: 12.5, marginTop: 8, lineHeight: 1.6 }}>
            A broad spectrum (large Δα = {result.fractal.multifractalWidth.toFixed(3)}) indicates
            richer, more heterogeneous texture — often associated with pathological tissue.
          </p>
        </Section>
        <Section title="Lacunarity across scales" delay={0.2}>
          <LacunarityBars fractal={result.fractal} />
        </Section>
      </div>

      {result.meta.notes && (
        <p className="muted" style={{ fontSize: 12.5, textAlign: "center" }}>{result.meta.notes}</p>
      )}
    </div>
  );
}

function ImpressionBanner({ result }: { result: AnalysisResponse }) {
  const severity = result.significant ? result.topFinding?.severity ?? "moderate" : "low";
  const color = severity === "high" ? "var(--bad)" : severity === "moderate" ? "var(--warn)" : "var(--accent-2)";
  return (
    <div
      style={{
        display: "flex",
        alignItems: "flex-start",
        gap: 10,
        padding: "12px 14px",
        borderRadius: 10,
        background: `${color}1a`,
        border: `1px solid ${color}55`,
      }}
    >
      <span style={{ color, fontSize: 15, lineHeight: 1.4 }}>{result.significant ? "●" : "○"}</span>
      <p style={{ fontSize: 13.5, lineHeight: 1.6, margin: 0, color: "var(--text-1)" }}>{result.impression}</p>
    </div>
  );
}

function LacunarityBars({ fractal }: { fractal: AnalysisResponse["fractal"] }) {
  const pts = fractal.lacunarityCurve;
  const max = Math.max(...pts.map((p) => p.value), 1.001);
  return (
    <div style={{ display: "flex", alignItems: "flex-end", gap: 14, height: 200, paddingTop: 10 }}>
      {pts.map((p, i) => (
        <div key={p.boxSize} style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", gap: 8, height: "100%", justifyContent: "flex-end" }}>
          <span className="mono muted" style={{ fontSize: 11 }}>{p.value.toFixed(2)}</span>
          <motion.div
            initial={{ height: 0 }}
            animate={{ height: `${(p.value / max) * 100}%` }}
            transition={{ delay: 0.25 + i * 0.08, duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
            style={{ width: "100%", borderRadius: "6px 6px 2px 2px", background: "linear-gradient(180deg, var(--accent), var(--accent-2))", minHeight: 4 }}
          />
          <span className="muted" style={{ fontSize: 11 }}>r={p.boxSize}</span>
        </div>
      ))}
    </div>
  );
}
