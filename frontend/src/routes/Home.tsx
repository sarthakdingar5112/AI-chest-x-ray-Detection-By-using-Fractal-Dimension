import { Link } from "react-router-dom";
import { motion } from "framer-motion";

const features = [
  {
    title: "Local FD heatmap",
    body: "A sliding-window fractal-dimension map that shows exactly where tissue texture becomes abnormally complex.",
  },
  {
    title: "Multi-disease CNN",
    body: "A DenseNet pretrained across NIH, CheXpert, MIMIC & PadChest scores 14+ pathologies per image.",
  },
  {
    title: "Hybrid fusion",
    body: "Fractal features (FD, lacunarity, multifractal spectrum) are fused with CNN embeddings for a sharper decision.",
  },
  {
    title: "Dual explainability",
    body: "Compare where the network looks (Grad-CAM) against where the texture is anomalous (fractal map).",
  },
];

export default function Home() {
  return (
    <div className="container" style={{ paddingBottom: 40 }}>
      <section style={{ textAlign: "center", padding: "70px 0 40px", position: "relative" }}>
        <motion.span
          className="badge"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          style={{ marginBottom: 22 }}
        >
          ● BE Final-Year Project · Medical Imaging × Fractal Geometry
        </motion.span>
        <motion.h1
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.05 }}
          style={{ fontSize: "clamp(34px, 6vw, 62px)", lineHeight: 1.05, maxWidth: 900, margin: "0 auto" }}
        >
          Detecting X-ray abnormalities through{" "}
          <span className="accent-text">fractal dimension analysis</span>
        </motion.h1>
        <motion.p
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.12 }}
          className="dim"
          style={{ maxWidth: 620, margin: "22px auto 0", fontSize: 17, lineHeight: 1.6 }}
        >
          Disease reshapes the texture of tissue. FractalScan quantifies that texture — its
          complexity, gappiness and multifractal structure — and fuses it with a deep
          network for explainable, multi-disease detection.
        </motion.p>
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          style={{ display: "flex", gap: 12, justifyContent: "center", marginTop: 34 }}
        >
          <Link to="/analyze" className="btn">Try the analyzer →</Link>
          <Link to="/methodology" className="btn" style={{ background: "transparent" }}>
            How it works
          </Link>
        </motion.div>
      </section>

      <section style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(230px, 1fr))", gap: 16, marginTop: 30 }}>
        {features.map((f, i) => (
          <motion.div
            key={f.title}
            className="card"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: i * 0.08 }}
            whileHover={{ y: -4 }}
            style={{ padding: 22 }}
          >
            <div style={{ width: 34, height: 34, borderRadius: 10, background: "linear-gradient(135deg, var(--accent), var(--accent-2))", marginBottom: 14, opacity: 0.9 }} />
            <h3 style={{ fontSize: 16.5 }}>{f.title}</h3>
            <p className="muted" style={{ marginTop: 8, fontSize: 13.5, lineHeight: 1.6 }}>{f.body}</p>
          </motion.div>
        ))}
      </section>
    </div>
  );
}
