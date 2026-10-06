import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import type { Overlays } from "../types/analysis";

type Layer = "fd" | "gradcam" | "lung";

export default function HeatmapViewer({ overlays }: { overlays: Overlays }) {
  const [layer, setLayer] = useState<Layer>("fd");
  const [opacity, setOpacity] = useState(0.75);

  const overlaySrc =
    layer === "fd" ? overlays.localFdMap : layer === "gradcam" ? overlays.gradcam : overlays.lungMask;
  const gradcamAvailable = !!overlays.gradcam;
  const lungAvailable = !!overlays.lungMask;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <div style={{ position: "relative", borderRadius: "var(--radius)", overflow: "hidden", border: "1px solid var(--border)" }}>
        <img src={overlays.original} alt="x-ray" style={{ width: "100%", display: "block" }} />
        <AnimatePresence mode="wait">
          {overlaySrc && (
            <motion.img
              key={layer}
              src={overlaySrc}
              alt={layer}
              initial={{ opacity: 0 }}
              animate={{ opacity }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.4 }}
              style={{
                position: "absolute",
                inset: 0,
                width: "100%",
                height: "100%",
                objectFit: "cover",
                mixBlendMode: layer === "lung" ? "normal" : "screen",
              }}
            />
          )}
        </AnimatePresence>
        <div style={{ position: "absolute", top: 12, left: 12, display: "flex", gap: 6, flexWrap: "wrap" }}>
          <Toggle active={layer === "fd"} onClick={() => setLayer("fd")}>Fractal texture</Toggle>
          <Toggle
            active={layer === "gradcam"}
            disabled={!gradcamAvailable}
            onClick={() => gradcamAvailable && setLayer("gradcam")}
          >
            CNN attention
          </Toggle>
          <Toggle
            active={layer === "lung"}
            disabled={!lungAvailable}
            onClick={() => lungAvailable && setLayer("lung")}
          >
            Lung fields
          </Toggle>
        </div>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <span className="muted" style={{ fontSize: 12.5, width: 66 }}>Overlay</span>
        <input
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={opacity}
          onChange={(e) => setOpacity(parseFloat(e.target.value))}
          style={{ flex: 1, accentColor: "var(--accent)" }}
        />
        <span className="mono muted" style={{ fontSize: 12.5, width: 40, textAlign: "right" }}>
          {Math.round(opacity * 100)}%
        </span>
      </div>

      <ColorLegend layer={layer} />

      <p className="muted" style={{ fontSize: 12.5, lineHeight: 1.6, margin: 0 }}>
        {layer === "fd"
          ? "Local fractal-dimension map — warmer regions have higher texture complexity, flagging where the tissue pattern deviates from healthy parenchyma. When lungs are segmented, the map is restricted to the lung fields."
          : layer === "gradcam"
          ? "CNN attention (Grad-CAM) — where the neural network focused when forming its prediction."
          : "Segmented lung fields — the region the fractal analysis is focused on, isolating parenchyma from ribs, heart and borders."}
      </p>
    </div>
  );
}

// The turbo colormap used by the backend heatmaps (blue = low → red = high).
const TURBO_GRADIENT =
  "linear-gradient(90deg, rgb(48,18,59), rgb(50,121,209), rgb(30,173,158), rgb(116,205,74), rgb(227,205,46), rgb(246,143,42), rgb(216,63,34), rgb(122,4,3))";

function ColorLegend({ layer }: { layer: Layer }) {
  // The lung layer is a simple outline, not a scale — show a swatch instead of a gradient.
  if (layer === "lung") {
    return (
      <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12 }}>
        <span style={{ width: 16, height: 12, borderRadius: 3, background: "rgba(53,224,212,0.55)", border: "1px solid #78f5eb", display: "inline-block" }} />
        <span className="muted">Highlighted area = detected lung region (where the fractal analysis runs)</span>
      </div>
    );
  }

  const ends =
    layer === "fd"
      ? { low: "Low complexity · smooth / normal", high: "High complexity · rough / suspicious" }
      : { low: "AI ignored", high: "AI focused here" };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
      <div
        style={{
          height: 12,
          borderRadius: 6,
          background: TURBO_GRADIENT,
          border: "1px solid var(--border)",
        }}
      />
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11.5 }}>
        <span className="muted">◀ {ends.low}</span>
        <span className="muted">{ends.high} ▶</span>
      </div>
    </div>
  );
}

function Toggle({
  active,
  disabled,
  onClick,
  children,
}: {
  active: boolean;
  disabled?: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className="badge"
      style={{
        cursor: disabled ? "not-allowed" : "pointer",
        opacity: disabled ? 0.4 : 1,
        background: active ? "rgba(53,224,212,0.18)" : "rgba(7,11,18,0.6)",
        borderColor: active ? "var(--accent)" : "var(--border)",
        color: active ? "var(--text-0)" : "var(--text-1)",
        backdropFilter: "blur(6px)",
      }}
    >
      {children}
    </button>
  );
}
