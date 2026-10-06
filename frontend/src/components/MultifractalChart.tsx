import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { SpectrumPoint } from "../types/analysis";

export default function MultifractalChart({ spectrum }: { spectrum: SpectrumPoint[] }) {
  const data = spectrum
    .filter((p) => Number.isFinite(p.alpha) && Number.isFinite(p.falpha))
    .map((p) => ({ alpha: Number(p.alpha.toFixed(4)), falpha: Number(p.falpha.toFixed(4)) }))
    .sort((a, b) => a.alpha - b.alpha);

  if (data.length < 3) {
    return <div className="muted" style={{ fontSize: 13 }}>Spectrum unavailable for this image.</div>;
  }

  return (
    <ResponsiveContainer width="100%" height={240}>
      <LineChart data={data} margin={{ top: 8, right: 12, bottom: 18, left: -8 }}>
        <CartesianGrid stroke="rgba(120,160,220,0.1)" strokeDasharray="3 3" />
        <XAxis
          dataKey="alpha"
          type="number"
          domain={["dataMin", "dataMax"]}
          stroke="var(--text-2)"
          tick={{ fontSize: 11 }}
          label={{ value: "α (singularity strength)", position: "bottom", fill: "var(--text-2)", fontSize: 11 }}
        />
        <YAxis stroke="var(--text-2)" tick={{ fontSize: 11 }} label={{ value: "f(α)", angle: -90, position: "insideLeft", fill: "var(--text-2)", fontSize: 11 }} />
        <Tooltip
          contentStyle={{
            background: "var(--surface-solid)",
            border: "1px solid var(--border-strong)",
            borderRadius: 10,
            fontSize: 12,
          }}
          labelStyle={{ color: "var(--text-1)" }}
        />
        <Line
          type="monotone"
          dataKey="falpha"
          stroke="var(--accent)"
          strokeWidth={2.4}
          dot={{ r: 2.5, fill: "var(--accent-2)" }}
          activeDot={{ r: 5 }}
          isAnimationActive
          animationDuration={1100}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
