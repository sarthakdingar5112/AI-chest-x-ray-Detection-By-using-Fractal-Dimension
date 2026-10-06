import { motion } from "framer-motion";
import type { Prediction } from "../types/analysis";

function barColor(p: number): string {
  if (p >= 0.6) return "var(--bad)";
  if (p >= 0.35) return "var(--warn)";
  return "var(--accent-2)";
}

export default function PredictionList({ predictions }: { predictions: Prediction[] }) {
  if (predictions.length === 0) {
    return (
      <div className="muted" style={{ fontSize: 14, lineHeight: 1.6 }}>
        No CNN predictions available. The ML model (TorchXRayVision) is not loaded on the
        server — the fractal analysis on the right is fully computed. Install
        <span className="mono"> torch + torchxrayvision</span> to enable multi-disease
        predictions.
      </div>
    );
  }
  const top = predictions.slice(0, 8);
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      {top.map((p, i) => (
        <div key={p.label}>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13.5, marginBottom: 5 }}>
            <span className="dim">{p.label}</span>
            <span className="mono" style={{ color: barColor(p.probability) }}>
              {(p.probability * 100).toFixed(1)}%
            </span>
          </div>
          <div style={{ height: 8, borderRadius: 999, background: "rgba(255,255,255,0.06)", overflow: "hidden" }}>
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${Math.min(p.probability * 100, 100)}%` }}
              transition={{ duration: 0.8, delay: i * 0.06, ease: [0.22, 1, 0.36, 1] }}
              style={{
                height: "100%",
                borderRadius: 999,
                background: `linear-gradient(90deg, ${barColor(p.probability)}, ${barColor(p.probability)}aa)`,
              }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}
