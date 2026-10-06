import { useEffect, useState } from "react";
import { motion } from "framer-motion";

type Row = Record<string, string | number>;
type Summary = {
  n_normal: number; n_pneumonia: number; n_covid_reference: number;
  significant_fractal_features: number; total_fractal_features: number;
  best_single_feature: string; best_single_feature_auc: number;
  rf_fractal_only: { AUC: number; accuracy: number; sensitivity: number; specificity: number };
  rf_brightness_only: { AUC: number };
  rf_fractal_plus_brightness: { AUC: number; accuracy: number };
};

const base = "/dataset-study/";

const figs = [
  { f: "fig3_example_xrays.png", t: "1 · What it looks like: normal vs diseased lungs",
    d: "Real chest X-rays from the dataset (typical cases) beside their local fractal-dimension maps. Warm colours = rougher, more complex texture. Diseased lungs show broader and stronger complexity." },
  { f: "fig2_mean_fd_maps.png", t: "2 · The average structure of each group",
    d: "Average local-FD map over 500 normal and 500 pneumonia X-rays. The right panel is the difference; dots mark regions where the difference is statistically significant." },
  { f: "fig1_feature_distributions.png", t: "3 · The feature distributions",
    d: "Box plots of the fractal features. Grayscale surface FD (DBC) and local FD rise clearly in pneumonia." },
  { f: "fig4_loglog_box_counting.png", t: "4 · Where the number comes from",
    d: "Class-average log-log plots. The slope of each line is the fractal dimension." },
  { f: "fig5_roc_and_importance.png", t: "5 · Do fractal features actually separate the groups?",
    d: "Patient-grouped 5-fold cross-validation. Fractal features alone beat a brightness-only control, and they add to it when combined." },
];

export default function DatasetStudy() {
  const [s, setS] = useState<Summary | null>(null);
  const [rows, setRows] = useState<Row[]>([]);
  const [fu, setFu] = useState<any>(null);
  const [md, setMd] = useState<Row[]>([]);
  useEffect(() => {
    fetch(base + "summary.json").then((r) => r.json()).then(setS).catch(() => {});
    fetch(base + "multi/dbc_table.json").then((r) => r.json()).then(setMd).catch(() => {});
    fetch(base + "fusion_results.json").then((r) => r.json()).then(setFu).catch(() => {});
    fetch(base + "table.json").then((r) => r.json()).then(setRows).catch(() => {});
  }, []);

  return (
    <div className="container" style={{ paddingBottom: 40 }}>
      <section style={{ maxWidth: 780, margin: "40px auto 10px", textAlign: "center" }}>
        <h1 style={{ fontSize: "clamp(28px, 5vw, 44px)" }}>Dataset study: healthy vs diseased structure</h1>
        <p className="dim" style={{ marginTop: 14, lineHeight: 1.65 }}>
          We ran the fractal analysis over a labelled public dataset to measure how lung texture
          changes when disease is present — not just on one uploaded image.
        </p>
      </section>

      {s && (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))", gap: 14, marginTop: 24 }}>
          {[
            ["Images analysed", `${s.n_normal} normal + ${s.n_pneumonia} pneumonia`],
            ["Significant fractal features", `${s.significant_fractal_features} of ${s.total_fractal_features}`],
            ["Best single feature", `${s.best_single_feature} (AUC ${s.best_single_feature_auc})`],
            ["Fractal-only classifier AUC", `${s.rf_fractal_only.AUC} (brightness-only: ${s.rf_brightness_only.AUC})`],
          ].map(([k, v]) => (
            <motion.div key={k} className="card" initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} style={{ padding: 18 }}>
              <div className="muted" style={{ fontSize: 12.5 }}>{k}</div>
              <div className="accent-text" style={{ fontSize: 17, fontWeight: 700, marginTop: 6 }}>{v}</div>
            </motion.div>
          ))}
        </div>
      )}

      {figs.map((x) => (
        <section key={x.f} className="card" style={{ padding: 22, marginTop: 22 }}>
          <h3 style={{ fontSize: 16 }}>{x.t}</h3>
          <p className="muted" style={{ fontSize: 13.5, margin: "6px 0 14px", lineHeight: 1.55 }}>{x.d}</p>
          <div style={{ background: "#fff", borderRadius: 12, padding: 10 }}>
            <img src={base + x.f} alt={x.t} style={{ width: "100%", display: "block" }} />
          </div>
        </section>
      ))}

      {rows.length > 0 && (
        <section className="card" style={{ padding: 22, marginTop: 22, overflowX: "auto" }}>
          <h3 style={{ fontSize: 16, marginBottom: 12 }}>Statistics table (Normal vs Pneumonia)</h3>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
            <thead>
              <tr className="muted" style={{ textAlign: "left" }}>
                {["Feature", "Normal", "Pneumonia", "Change", "Cohen d", "AUC", "Significant"].map((h) => (
                  <th key={h} style={{ padding: "6px 8px" }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={String(r.key)} style={{ borderTop: "1px solid var(--border)" }}>
                  <td style={{ padding: "7px 8px" }}>{r.feature}</td>
                  <td className="mono">{r["Normal (mean ± SD)"]}</td>
                  <td className="mono">{r["Pneumonia (mean ± SD)"]}</td>
                  <td className="mono">{Number(r["% change"]) > 0 ? "+" : ""}{Number(r["% change"]).toFixed(1)}%</td>
                  <td className="mono">{Number(r["Cohen d"]).toFixed(2)}</td>
                  <td className="mono">{Number(r["AUC (single feature)"]).toFixed(2)}</td>
                  <td style={{ color: r.significant === "yes" ? "var(--good)" : "var(--text-2)" }}>{r.significant}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      {fu && (
        <section className="card" style={{ padding: 22, marginTop: 22 }}>
          <h3 style={{ fontSize: 16 }}>6 · Does fractal add value on top of the AI? (cross-validated, {fu.n_images} images)</h3>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13.5, marginTop: 12 }}>
            <tbody>
              {[
                ["Pretrained AI, zero-shot", fu.A_cnn_zero_shot.AUC, null],
                ["Fractal features only", fu.fractal_only_mlp.AUC, fu.fractal_only_mlp.accuracy],
                ["AI embedding + trained head", fu.B_cnn_head.AUC, fu.B_cnn_head.accuracy],
                ["AI embedding + fractal (fusion)", fu.C_cnn_plus_fractal.AUC, fu.C_cnn_plus_fractal.accuracy],
              ].map(([k, a, acc]: any) => (
                <tr key={k} style={{ borderTop: "1px solid var(--border)" }}>
                  <td style={{ padding: "8px" }}>{k}</td>
                  <td className="mono accent-text">AUC {a.toFixed(3)}</td>
                  <td className="mono muted">{acc ? `accuracy ${(acc * 100).toFixed(1)}%` : ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="muted" style={{ fontSize: 13.5, marginTop: 12, lineHeight: 1.6 }}>
            Fractal features alone beat the untrained AI, so they carry real information. Once the AI is trained,
            adding them gives no extra accuracy (gain {fu.AUC_gain_C_minus_B.mean.toFixed(3)}, 95% CI {fu.AUC_gain_C_minus_B.ci95[0].toFixed(3)} to {fu.AUC_gain_C_minus_B.ci95[1].toFixed(3)}).
            Their value here is interpretability: the heatmaps and the measurable healthy-vs-diseased texture difference.
          </p>
          <p className="muted" style={{ fontSize: 13.5, marginTop: 10, lineHeight: 1.6 }}>
            <strong>External-validity check:</strong> this dataset-trained head looked excellent in cross-validation but over-called
            pneumonia on chest X-rays from other sources (dataset shift). So the live app uses the pretrained multi-hospital
            model plus fractal analysis; the trained head is reported here as an experiment.
          </p>
        </section>
      )}


      <section style={{ maxWidth: 820, margin: "46px auto 0" }}>
        <h2 style={{ fontSize: "clamp(22px, 3.5vw, 30px)" }}>Part 2 · Various diseases: how does each one look?</h2>
        <p className="dim" style={{ marginTop: 10, lineHeight: 1.65 }}>
          We repeated the study for several conditions. Each disease is compared only with normal X-rays from the
          <strong> same source</strong>, because different hospitals and scanners change texture numbers on their own.
        </p>
      </section>

      {[
        { f: "multi/fig9_examples_per_disease.png", t: "A · A typical X-ray of each condition and its fractal texture map",
          d: "Warm colours = rougher, more complex texture. Different diseases distribute complexity differently across the lungs." },
        { f: "multi/fig6_effect_sizes_per_disease.png", t: "B · The fractal 'signature' of each disease",
          d: "Each cell is the effect size (Cohen d) of a fractal feature: red = higher than normal, blue = lower. * = statistically significant. Rows with only 15 normals are exploratory." },
        { f: "multi/fig8_difference_maps_per_disease.png", t: "C · Where in the lungs the texture changes",
          d: "Average local-FD map of the disease minus the average of its own normal group." },
        { f: "multi/fig7_dbc_fd_per_disease.png", t: "D · Grayscale fractal dimension per disease",
          d: "Compare boxes only inside the same panel (same source)." },
      ].map((x) => (
        <section key={x.f} className="card" style={{ padding: 22, marginTop: 22 }}>
          <h3 style={{ fontSize: 16 }}>{x.t}</h3>
          <p className="muted" style={{ fontSize: 13.5, margin: "6px 0 14px", lineHeight: 1.55 }}>{x.d}</p>
          <div style={{ background: "#fff", borderRadius: 12, padding: 10 }}>
            <img src={base + x.f} alt={x.t} style={{ width: "100%", display: "block" }} />
          </div>
        </section>
      ))}

      {md.length > 0 && (
        <section className="card" style={{ padding: 22, marginTop: 22, overflowX: "auto" }}>
          <h3 style={{ fontSize: 16, marginBottom: 12 }}>Grayscale fractal dimension (DBC): disease vs its own normal group</h3>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
            <thead>
              <tr className="muted" style={{ textAlign: "left" }}>
                {["Disease", "n (disease / normal)", "Normal", "Disease", "Change", "Cohen d", "Significant"].map((h) => (
                  <th key={h} style={{ padding: "6px 8px" }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {md.map((r, i) => (
                <tr key={i} style={{ borderTop: "1px solid var(--border)" }}>
                  <td style={{ padding: "7px 8px" }}>{r.disease} <span className="muted">[{r.source}]</span></td>
                  <td className="mono">{r.n_disease} / {r.n_normal}</td>
                  <td className="mono">{Number(r.normal_mean).toFixed(3)}</td>
                  <td className="mono">{Number(r.disease_mean).toFixed(3)}</td>
                  <td className="mono">{Number(r.pct_change) > 0 ? "+" : ""}{Number(r.pct_change).toFixed(1)}%</td>
                  <td className="mono">{Number(r.cohen_d).toFixed(2)}</td>
                  <td style={{ color: r.significant ? "var(--good)" : "var(--text-2)" }}>{r.significant ? "yes" : "no (too few cases)"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      <section className="card" style={{ padding: 22, marginTop: 22 }}>
        <h3 style={{ fontSize: 16 }}>What this shows (and does not show)</h3>
        <ul className="muted" style={{ fontSize: 13.5, lineHeight: 1.7, marginTop: 8, paddingLeft: 18 }}>
          <li><strong>Pneumonia</strong> (500 vs 500): clearly more complex texture (higher fractal dimension, large effect).</li>
          <li><strong>COVID-19</strong> (712 vs 800): a <em>different</em> signature: lower surface FD, higher lacunarity, lower multifractal width, changes concentrated toward the lung periphery.</li>
          <li><strong>TB, bacterial, viral, fungal</strong>: only 11–49 cases and 15 normals in the public set; no difference reaches significance, so we cannot conclude anything yet. These are exploratory.</li>
          <li>Takeaway: diseases do not all move texture in the same direction. Each has its own fractal signature, and the strongest evidence so far is for pneumonia and COVID-19.</li>
          <li>Source differences (scanner, compression) can mimic disease effects, which is why every disease is compared only with same-source normals.</li>
        </ul>
      </section>

      <section className="card" style={{ padding: 22, marginTop: 22 }}>
        <h3 style={{ fontSize: 16 }}>Honest limitations</h3>
        <ul className="muted" style={{ fontSize: 13.5, lineHeight: 1.7, marginTop: 8, paddingLeft: 18 }}>
          <li>Dataset: public "COVID-19 / X-Ray Image DataSet" (Normal and Pneumonia images originate from NIH ChestX-ray14). Labels are image-level and not radiologist-verified for this study.</li>
          <li>Fractal features describe texture statistics; they are evidence, not a diagnosis. Differences are real on average but individual images overlap.</li>
          <li>Texture is measured on the central thorax region; scanner and exposure differences can affect values (a brightness-only control is reported for this reason).</li>
          <li>COVID-19 images come from different sources, so they are shown for reference only.</li>
        </ul>
      </section>
    </div>
  );
}
