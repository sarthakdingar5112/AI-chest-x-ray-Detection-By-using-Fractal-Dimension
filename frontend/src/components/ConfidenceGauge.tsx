import { motion } from "framer-motion";
import type { TopFinding } from "../types/analysis";

const sevColor: Record<string, string> = {
  high: "var(--bad)",
  moderate: "var(--warn)",
  low: "var(--good)",
};

export default function ConfidenceGauge({ finding }: { finding: TopFinding }) {
  const pct = Math.round(finding.probability * 100);
  const R = 62;
  const C = 2 * Math.PI * R;
  const color = sevColor[finding.severity] ?? "var(--accent)";

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 10 }}>
      <div style={{ position: "relative", width: 160, height: 160 }}>
        <svg width="160" height="160" style={{ transform: "rotate(-90deg)" }}>
          <circle cx="80" cy="80" r={R} stroke="rgba(255,255,255,0.07)" strokeWidth="12" fill="none" />
          <motion.circle
            cx="80"
            cy="80"
            r={R}
            stroke={color}
            strokeWidth="12"
            fill="none"
            strokeLinecap="round"
            strokeDasharray={C}
            initial={{ strokeDashoffset: C }}
            animate={{ strokeDashoffset: C * (1 - finding.probability) }}
            transition={{ duration: 1.1, ease: [0.22, 1, 0.36, 1] }}
            style={{ filter: `drop-shadow(0 0 8px ${color})` }}
          />
        </svg>
        <div
          style={{
            position: "absolute",
            inset: 0,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <CountUp value={pct} />
          <span className="muted" style={{ fontSize: 12 }}>confidence</span>
        </div>
      </div>
      <div style={{ textAlign: "center" }}>
        <div style={{ fontSize: 18, fontWeight: 650 }}>{finding.label}</div>
        <span className={`badge sev-${finding.severity}`} style={{ marginTop: 6 }}>
          {finding.severity.toUpperCase()} likelihood
        </span>
      </div>
    </div>
  );
}

function CountUp({ value }: { value: number }) {
  return (
    <motion.span
      style={{ fontSize: 40, fontWeight: 700 }}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
    >
      <motion.span
        initial={{ y: 8 }}
        animate={{ y: 0 }}
        transition={{ duration: 0.6 }}
      >
        {value}
      </motion.span>
      <span style={{ fontSize: 20, opacity: 0.6 }}>%</span>
    </motion.span>
  );
}
