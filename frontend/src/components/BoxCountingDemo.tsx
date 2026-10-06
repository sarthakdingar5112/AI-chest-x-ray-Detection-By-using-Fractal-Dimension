import { useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";

const SIZE = 324; // canvas px (multiple of many box sizes)
const BOX_SIZES = [4, 6, 9, 12, 18, 27, 36, 54];

/**
 * Interactive box-counting demo. Renders a Sierpinski triangle, lets the user change the
 * box size, and shows: the grid overlay with occupied boxes highlighted, the running count
 * N(s), and the fractal-dimension estimate from the log-log fit. Makes the core concept
 * click instantly for a viva audience.
 */
export default function BoxCountingDemo() {
  const [boxIdx, setBoxIdx] = useState(3);
  const boxSize = BOX_SIZES[boxIdx];
  const occupancy = useSierpinskiOccupancy(SIZE);

  const { count, occupied } = useMemo(() => countBoxes(occupancy, SIZE, boxSize), [occupancy, boxSize]);

  // Log-log points across all box sizes -> slope = fractal dimension.
  const { points, dimension } = useMemo(() => {
    const pts = BOX_SIZES.map((s) => {
      const c = countBoxes(occupancy, SIZE, s).count;
      return { logInvS: Math.log(1 / s), logN: Math.log(Math.max(c, 1)), s, c };
    });
    return { points: pts, dimension: slope(pts.map((p) => p.logInvS), pts.map((p) => p.logN)) };
  }, [occupancy]);

  return (
    <div style={{ display: "grid", gridTemplateColumns: "auto 1fr", gap: 26, alignItems: "center", flexWrap: "wrap" }}>
      <div>
        <div style={{ position: "relative", width: SIZE, height: SIZE, maxWidth: "100%" }}>
          <FractalCanvas occupancy={occupancy} />
          <GridOverlay boxSize={boxSize} occupied={occupied} />
        </div>
        <div style={{ marginTop: 16 }}>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12.5, marginBottom: 6 }}>
            <span className="muted">Box size s = {boxSize}px</span>
            <span className="mono">N(s) = {count}</span>
          </div>
          <input
            type="range"
            min={0}
            max={BOX_SIZES.length - 1}
            step={1}
            value={boxIdx}
            onChange={(e) => setBoxIdx(parseInt(e.target.value))}
            style={{ width: "100%", accentColor: "var(--accent)" }}
          />
        </div>
      </div>

      <div>
        <div className="card" style={{ padding: 18 }}>
          <LogLogPlot points={points} activeS={boxSize} />
        </div>
        <div style={{ marginTop: 16, display: "flex", alignItems: "baseline", gap: 10 }}>
          <span className="muted" style={{ fontSize: 13 }}>Estimated fractal dimension</span>
          <span className="mono accent-text" style={{ fontSize: 30, fontWeight: 700 }}>
            D ≈ {dimension.toFixed(3)}
          </span>
        </div>
        <p className="muted" style={{ fontSize: 12.5, lineHeight: 1.6, marginTop: 6 }}>
          As boxes shrink, the count of occupied boxes grows as N(s) ∝ s<sup>−D</sup>. The
          slope of log N against log(1/s) is the dimension. For this Sierpinski triangle the
          true value is log 3 / log 2 ≈ 1.585.
        </p>
      </div>
    </div>
  );
}

function FractalCanvas({ occupancy }: { occupancy: Uint8Array }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const ctx = ref.current?.getContext("2d");
    if (!ctx) return;
    const img = ctx.createImageData(SIZE, SIZE);
    for (let i = 0; i < SIZE * SIZE; i++) {
      const on = occupancy[i];
      img.data[i * 4] = on ? 53 : 10;
      img.data[i * 4 + 1] = on ? 224 : 16;
      img.data[i * 4 + 2] = on ? 212 : 26;
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [occupancy]);
  return <canvas ref={ref} width={SIZE} height={SIZE} style={{ borderRadius: 12, width: SIZE, height: SIZE, maxWidth: "100%" }} />;
}

function GridOverlay({ boxSize, occupied }: { boxSize: number; occupied: boolean[] }) {
  const n = Math.ceil(SIZE / boxSize);
  const cells = [];
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const on = occupied[i * n + j];
      cells.push(
        <div
          key={`${i}-${j}`}
          style={{
            position: "absolute",
            top: i * boxSize,
            left: j * boxSize,
            width: boxSize,
            height: boxSize,
            border: "1px solid rgba(255,255,255,0.10)",
            background: on ? "rgba(255,92,108,0.28)" : "transparent",
            boxShadow: on ? "inset 0 0 0 1px rgba(255,92,108,0.7)" : "none",
          }}
        />
      );
    }
  }
  return <div style={{ position: "absolute", inset: 0 }}>{cells}</div>;
}

function LogLogPlot({ points, activeS }: { points: { logInvS: number; logN: number; s: number }[]; activeS: number }) {
  const W = 300, H = 190, pad = 30;
  const xs = points.map((p) => p.logInvS);
  const ys = points.map((p) => p.logN);
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const ymin = Math.min(...ys), ymax = Math.max(...ys);
  const sx = (x: number) => pad + ((x - xmin) / (xmax - xmin || 1)) * (W - 2 * pad);
  const sy = (y: number) => H - pad - ((y - ymin) / (ymax - ymin || 1)) * (H - 2 * pad);
  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`}>
      <line x1={pad} y1={H - pad} x2={W - pad} y2={H - pad} stroke="var(--border-strong)" />
      <line x1={pad} y1={pad} x2={pad} y2={H - pad} stroke="var(--border-strong)" />
      <motion.polyline
        points={points.map((p) => `${sx(p.logInvS)},${sy(p.logN)}`).join(" ")}
        fill="none"
        stroke="var(--accent)"
        strokeWidth={2}
        initial={{ pathLength: 0 }}
        animate={{ pathLength: 1 }}
        transition={{ duration: 0.8 }}
      />
      {points.map((p) => (
        <circle key={p.s} cx={sx(p.logInvS)} cy={sy(p.logN)} r={p.s === activeS ? 5 : 3} fill={p.s === activeS ? "var(--bad)" : "var(--accent-2)"} />
      ))}
      <text x={W / 2} y={H - 6} fontSize="10" fill="var(--text-2)" textAnchor="middle">log(1/s)</text>
      <text x={10} y={H / 2} fontSize="10" fill="var(--text-2)" transform={`rotate(-90 10 ${H / 2})`} textAnchor="middle">log N(s)</text>
    </svg>
  );
}

// --- occupancy + counting helpers -----------------------------------------------------
function useSierpinskiOccupancy(size: number): Uint8Array {
  return useMemo(() => {
    const occ = new Uint8Array(size * size);
    // Chaos-game render of the Sierpinski triangle.
    const verts = [
      [size / 2, 8],
      [8, size - 8],
      [size - 8, size - 8],
    ];
    let x = size / 2, y = size / 2;
    for (let it = 0; it < 120000; it++) {
      const v = verts[(Math.random() * 3) | 0];
      x = (x + v[0]) / 2;
      y = (y + v[1]) / 2;
      if (it > 20) {
        const px = Math.floor(x), py = Math.floor(y);
        if (px >= 0 && px < size && py >= 0 && py < size) occ[py * size + px] = 1;
      }
    }
    return occ;
  }, [size]);
}

function countBoxes(occ: Uint8Array, size: number, boxSize: number) {
  const n = Math.ceil(size / boxSize);
  const occupied = new Array<boolean>(n * n).fill(false);
  for (let p = 0; p < size * size; p++) {
    if (!occ[p]) continue;
    const y = (p / size) | 0;
    const x = p % size;
    const bi = (y / boxSize) | 0;
    const bj = (x / boxSize) | 0;
    occupied[bi * n + bj] = true;
  }
  const count = occupied.reduce((a, b) => a + (b ? 1 : 0), 0);
  return { count, occupied };
}

function slope(xs: number[], ys: number[]): number {
  const n = xs.length;
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) {
    num += (xs[i] - mx) * (ys[i] - my);
    den += (xs[i] - mx) ** 2;
  }
  return den === 0 ? 0 : num / den;
}
