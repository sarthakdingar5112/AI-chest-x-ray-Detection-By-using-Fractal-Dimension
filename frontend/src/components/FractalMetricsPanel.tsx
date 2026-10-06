import { motion } from "framer-motion";
import type { FractalMetrics } from "../types/analysis";

interface Metric {
  label: string;
  value: number;
  hint: string;
  decimals?: number;
}

export default function FractalMetricsPanel({ fractal }: { fractal: FractalMetrics }) {
  const metrics: Metric[] = [
    { label: "Box-counting FD", value: fractal.boxCountingFD, hint: "structural complexity of the binarised image (1–2)" },
    { label: "DBC surface FD", value: fractal.dbcFD, hint: "grayscale texture roughness (2–3)" },
    { label: "Lacunarity", value: fractal.lacunarity, hint: "texture heterogeneity / gappiness (≥1)" },
    { label: "Multifractal width Δα", value: fractal.multifractalWidth, hint: "range of local scaling behaviours" },
    { label: "Local FD (mean)", value: fractal.localFdMean, hint: "average complexity across patches" },
    { label: "Local FD (σ)", value: fractal.localFdStd, hint: "spatial variability of complexity" },
  ];
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 12 }}>
      {metrics.map((m, i) => (
        <motion.div
          key={m.label}
          className="card"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: i * 0.05 }}
          style={{ padding: "16px 16px 14px" }}
        >
          <div className="mono" style={{ fontSize: 26, fontWeight: 700 }}>
            <AnimatedNumber value={m.value} decimals={m.decimals ?? 3} />
          </div>
          <div style={{ fontSize: 13, fontWeight: 600, marginTop: 2 }}>{m.label}</div>
          <div className="muted" style={{ fontSize: 11.5, marginTop: 4, lineHeight: 1.4 }}>{m.hint}</div>
        </motion.div>
      ))}
    </div>
  );
}

import { useEffect, useState } from "react";

function AnimatedNumber({ value, decimals }: { value: number; decimals: number }) {
  const [display, setDisplay] = useState(0);
  useEffect(() => {
    const start = performance.now();
    const dur = 900;
    let raf = 0;
    const tick = (t: number) => {
      const p = Math.min((t - start) / dur, 1);
      const eased = 1 - Math.pow(1 - p, 3);
      setDisplay(value * eased);
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [value]);
  return <>{display.toFixed(decimals)}</>;
}
