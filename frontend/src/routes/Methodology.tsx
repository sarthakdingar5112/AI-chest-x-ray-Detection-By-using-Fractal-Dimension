import { motion } from "framer-motion";
import BoxCountingDemo from "../components/BoxCountingDemo";

const steps = [
  { n: "01", t: "Preprocess", d: "Load PNG/JPG or DICOM, convert to grayscale, and apply CLAHE contrast enhancement to reveal texture." },
  { n: "02", t: "Fractal analysis", d: "Compute box-counting & differential-box-counting FD, lacunarity, the multifractal spectrum, and a local FD map." },
  { n: "03", t: "CNN inference", d: "A DenseNet pretrained on four large chest datasets outputs pathology probabilities and a 1024-d embedding." },
  { n: "04", t: "Fusion", d: "A small trained head combines the fractal feature vector with the CNN embedding for the final decision." },
  { n: "05", t: "Explain", d: "Render the local FD heatmap and Grad-CAM so the prediction is visually accountable." },
];

export default function Methodology() {
  return (
    <div className="container" style={{ paddingBottom: 40 }}>
      <section style={{ maxWidth: 720, margin: "40px auto 10px", textAlign: "center" }}>
        <h1 style={{ fontSize: "clamp(28px, 5vw, 44px)" }}>How fractal dimension detects abnormality</h1>
        <p className="dim" style={{ marginTop: 14, lineHeight: 1.65 }}>
          Fractal dimension measures how detail in a pattern changes with the scale at which
          you look. Healthy and diseased tissue scale differently — so the dimension becomes
          a fingerprint of texture. Try the box-counting method below.
        </p>
      </section>

      <motion.section
        className="card"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        style={{ padding: 26, marginTop: 24 }}
      >
        <h3 style={{ fontSize: 16, marginBottom: 4 }}>Interactive box-counting</h3>
        <p className="muted" style={{ fontSize: 13.5, marginBottom: 22 }}>
          Drag the slider to change the box size. Occupied boxes are highlighted; the log-log
          plot updates and its slope is the fractal dimension.
        </p>
        <BoxCountingDemo />
      </motion.section>

      <section style={{ marginTop: 30 }}>
        <h3 style={{ fontSize: 18, marginBottom: 16 }}>The pipeline, end to end</h3>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 14 }}>
          {steps.map((s, i) => (
            <motion.div
              key={s.n}
              className="card"
              initial={{ opacity: 0, y: 16 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.06 }}
              style={{ padding: 20 }}
            >
              <span className="mono accent-text" style={{ fontSize: 20, fontWeight: 700 }}>{s.n}</span>
              <h3 style={{ fontSize: 15.5, marginTop: 8 }}>{s.t}</h3>
              <p className="muted" style={{ fontSize: 13, marginTop: 6, lineHeight: 1.55 }}>{s.d}</p>
            </motion.div>
          ))}
        </div>
      </section>

      <section className="card" style={{ padding: 26, marginTop: 30 }}>
        <h3 style={{ fontSize: 16, marginBottom: 14 }}>The three texture measures we extract</h3>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: 18 }}>
          <Measure title="Fractal dimension (FD)" body="A single number for texture complexity. Higher FD = rougher, more detailed structure across scales." />
          <Measure title="Lacunarity" body="How gappy or heterogeneous the pattern is. Two textures with equal FD can differ sharply in lacunarity." />
          <Measure title="Multifractal spectrum f(α)" body="Not one dimension but a whole range — capturing how scaling varies locally. Its width Δα is highly discriminative." />
        </div>
      </section>
    </div>
  );
}

function Measure({ title, body }: { title: string; body: string }) {
  return (
    <div>
      <div style={{ width: 30, height: 4, borderRadius: 999, background: "linear-gradient(90deg, var(--accent), var(--accent-2))", marginBottom: 12 }} />
      <h4 style={{ fontSize: 14.5, fontWeight: 650 }}>{title}</h4>
      <p className="muted" style={{ fontSize: 13, marginTop: 6, lineHeight: 1.6 }}>{body}</p>
    </div>
  );
}
