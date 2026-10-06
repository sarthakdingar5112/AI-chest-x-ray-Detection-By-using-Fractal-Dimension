import { jsPDF } from "jspdf";
import type { AnalysisResponse } from "../types/analysis";

// Brand palette (RGB) for the printable report.
const CYAN: [number, number, number] = [30, 170, 165];
const BLUE: [number, number, number] = [58, 110, 200];
const INK: [number, number, number] = [28, 36, 52];
const MUTE: [number, number, number] = [110, 122, 140];
const LINE: [number, number, number] = [222, 228, 238];
const BAD: [number, number, number] = [214, 65, 82];
const WARN: [number, number, number] = [200, 150, 40];
const GOOD: [number, number, number] = [40, 170, 95];

const sevColor = (s: string) => (s === "high" ? BAD : s === "moderate" ? WARN : GOOD);

interface ImgDim { src: string; w: number; h: number; }

function loadImage(src: string): Promise<ImgDim> {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve({ src, w: img.naturalWidth, h: img.naturalHeight });
    img.onerror = () => resolve({ src, w: 1, h: 1 });
    img.src = src;
  });
}

const metricInterpretation = (r: AnalysisResponse): [string, string, string][] => [
  ["Box-counting FD", r.fractal.boxCountingFD.toFixed(3), "Structural complexity of the binarised image (1-2)."],
  ["DBC surface FD", r.fractal.dbcFD.toFixed(3), "Grayscale texture roughness of the intensity surface (2-3)."],
  ["Lacunarity", r.fractal.lacunarity.toFixed(3), "Texture heterogeneity / gappiness (>= 1; higher = more clustered)."],
  ["Multifractal width", r.fractal.multifractalWidth.toFixed(3), "Range of local scaling behaviours; broader = richer texture."],
  ["Local FD (mean)", r.fractal.localFdMean.toFixed(3), "Average patch-wise complexity across the image."],
  ["Local FD (std)", r.fractal.localFdStd.toFixed(3), "Spatial variability of complexity; higher = more focal change."],
];

export async function generateReport(result: AnalysisResponse): Promise<void> {
  const doc = new jsPDF({ unit: "pt", format: "a4" });
  const W = doc.internal.pageSize.getWidth();
  const M = 42; // margin
  let y = 0;

  // ---- Header band --------------------------------------------------------------
  doc.setFillColor(...INK);
  doc.rect(0, 0, W, 78, "F");
  doc.setFillColor(...CYAN);
  doc.rect(0, 78, W, 3, "F");
  doc.setTextColor(255, 255, 255);
  doc.setFont("helvetica", "bold");
  doc.setFontSize(20);
  doc.text("FractalScan", M, 38);
  doc.setFont("helvetica", "normal");
  doc.setFontSize(11);
  doc.setTextColor(180, 200, 230);
  doc.text("X-Ray Abnormality Analysis Report", M, 56);
  doc.setFontSize(9);
  const now = new Date();
  doc.text(`Generated ${now.toLocaleString()}`, W - M, 38, { align: "right" });
  doc.text(`Model: ${result.meta.model}`, W - M, 52, { align: "right" });
  doc.text(`Processing: ${result.meta.processingMs} ms`, W - M, 66, { align: "right" });

  y = 108;

  // ---- Primary finding ----------------------------------------------------------
  doc.setTextColor(...MUTE);
  doc.setFontSize(9);
  doc.text("PRIMARY FINDING", M, y);
  y += 20;
  if (result.topFinding) {
    const tf = result.topFinding;
    doc.setTextColor(...INK);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(22);
    doc.text(tf.label, M, y);
    // confidence pill
    const pct = `${Math.round(tf.probability * 100)}%`;
    const [r, g, b] = sevColor(tf.severity);
    doc.setFillColor(r, g, b);
    doc.roundedRect(W - M - 150, y - 18, 150, 26, 6, 6, "F");
    doc.setTextColor(255, 255, 255);
    doc.setFontSize(12);
    doc.text(`${pct}  ${tf.severity.toUpperCase()}`, W - M - 75, y - 1, { align: "center" });
    y += 16;
  } else {
    doc.setTextColor(...MUTE);
    doc.setFont("helvetica", "italic");
    doc.setFontSize(12);
    doc.text("No CNN prediction available (fractal-only analysis).", M, y);
    y += 8;
  }
  y += 18;

  // ---- Predictions table --------------------------------------------------------
  if (result.predictions.length > 0) {
    y = sectionTitle(doc, "Pathology probabilities", M, y);
    const top = result.predictions.slice(0, 8);
    doc.setFontSize(10);
    for (const p of top) {
      doc.setTextColor(...INK);
      doc.setFont("helvetica", "normal");
      doc.text(p.label, M, y);
      // bar
      const barX = M + 200, barW = W - M - barX - 52;
      doc.setFillColor(...LINE);
      doc.roundedRect(barX, y - 8, barW, 8, 4, 4, "F");
      const col = p.probability >= 0.6 ? BAD : p.probability >= 0.35 ? WARN : BLUE;
      doc.setFillColor(...col);
      doc.roundedRect(barX, y - 8, Math.max(barW * Math.min(p.probability, 1), 2), 8, 4, 4, "F");
      doc.setTextColor(...MUTE);
      doc.text(`${(p.probability * 100).toFixed(1)}%`, W - M, y, { align: "right" });
      y += 20;
    }
    y += 8;
  }

  // ---- Fractal signature table --------------------------------------------------
  y = sectionTitle(doc, "Fractal signature", M, y);
  doc.setFontSize(9.5);
  for (const [name, val, desc] of metricInterpretation(result)) {
    doc.setTextColor(...INK);
    doc.setFont("helvetica", "bold");
    doc.text(name, M, y);
    doc.setFont("helvetica", "normal");
    doc.setTextColor(...BLUE);
    doc.text(val, M + 130, y);
    doc.setTextColor(...MUTE);
    doc.text(desc, M + 185, y);
    y += 17;
  }
  y += 6;

  // ---- Images (page 2) ----------------------------------------------------------
  const imgs: { label: string; src: string | null }[] = [
    { label: "Original (CLAHE)", src: result.overlays.original },
    { label: result.meta.segmentation ? "Segmented lung fields" : "Lung fields (n/a)", src: result.overlays.lungMask },
    { label: "Local FD heatmap", src: result.overlays.localFdMap },
    { label: "CNN attention (Grad-CAM)", src: result.overlays.gradcam },
  ];
  const loaded = await Promise.all(imgs.filter((i) => i.src).map(async (i) => ({ label: i.label, dim: await loadImage(i.src!) })));

  doc.addPage();
  y = 48;
  doc.setTextColor(...INK);
  doc.setFont("helvetica", "bold");
  doc.setFontSize(14);
  doc.text("Imaging & explainability", M, y);
  y += 8;
  doc.setDrawColor(...LINE);
  doc.line(M, y, W - M, y);
  y += 24;

  const gap = 16;
  const cols = loaded.length === 4 ? 2 : Math.min(loaded.length, 3);
  const cellW = (W - 2 * M - (cols - 1) * gap) / cols;
  const rowH = cols >= 3 ? 200 : 230; // image cap per row
  let maxRowH = 0;
  loaded.forEach(({ label, dim }, i) => {
    const col = i % cols;
    const row = Math.floor(i / cols);
    const x = M + col * (cellW + gap);
    const cy = y + row * (rowH + 34);
    const scale = Math.min(cellW / dim.w, rowH / dim.h);
    const drawW = dim.w * scale;
    const drawH = dim.h * scale;
    doc.addImage(dim.src, "PNG", x, cy, drawW, drawH);
    doc.setDrawColor(...LINE);
    doc.rect(x, cy, drawW, drawH);
    doc.setTextColor(...MUTE);
    doc.setFont("helvetica", "normal");
    doc.setFontSize(8.5);
    doc.text(label, x, cy + drawH + 14);
    maxRowH = Math.max(maxRowH, (row + 1) * (rowH + 34));
  });
  y += maxRowH + 16;

  // Heatmap color legend so the reader knows what the colors mean.
  drawColorLegend(doc, M, y, 200);
  y += 34;

  doc.setTextColor(...MUTE);
  doc.setFontSize(9);
  const caption =
    "The local FD heatmap shows where tissue-texture complexity deviates from healthy parenchyma; " +
    "Grad-CAM shows where the CNN focused. Regions where both agree are the strongest evidence.";
  const capLines = doc.splitTextToSize(caption, W - 2 * M) as string[];
  doc.text(capLines, M, y);
  y += capLines.length * 12 + 28;

  // ---- Multifractal spectrum (native vector plot) -------------------------------
  const CHART_H = 200;
  if (y + CHART_H + 60 > doc.internal.pageSize.getHeight() - 50) {
    doc.addPage();
    y = 48;
  }
  drawSpectrumChart(doc, result, M, y, W - 2 * M, CHART_H);

  // ---- Footer on both pages -----------------------------------------------------
  const pages = doc.getNumberOfPages();
  for (let p = 1; p <= pages; p++) {
    doc.setPage(p);
    const H = doc.internal.pageSize.getHeight();
    doc.setDrawColor(...LINE);
    doc.line(M, H - 40, W - M, H - 40);
    doc.setTextColor(...MUTE);
    doc.setFontSize(8);
    doc.text("FractalScan - BE Final-Year Project. Research/education use only; not a medical device.", M, H - 26);
    doc.text(`Page ${p} of ${pages}`, W - M, H - 26, { align: "right" });
  }

  doc.save(`fractalscan-report-${now.toISOString().slice(0, 19).replace(/[:T]/g, "-")}.pdf`);
}

// Turbo colormap anchors (must match backend app/utils/images.py).
const TURBO_ANCHORS: [number, number, number][] = [
  [48, 18, 59], [50, 121, 209], [30, 173, 158], [116, 205, 74],
  [227, 205, 46], [246, 143, 42], [216, 63, 34], [122, 4, 3],
];

function turboAt(t: number): [number, number, number] {
  const x = Math.max(0, Math.min(1, t)) * (TURBO_ANCHORS.length - 1);
  const i = Math.min(Math.floor(x), TURBO_ANCHORS.length - 2);
  const f = x - i;
  const a = TURBO_ANCHORS[i], b = TURBO_ANCHORS[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

/** Draw the heatmap color scale (blue = low, red = high) as thin strips, with labels. */
function drawColorLegend(doc: jsPDF, x: number, y: number, w: number): void {
  doc.setTextColor(...INK);
  doc.setFont("helvetica", "bold");
  doc.setFontSize(9);
  doc.text("Heatmap color scale", x, y);
  const barY = y + 6;
  const barH = 9;
  const strips = 60;
  const sw = w / strips;
  for (let i = 0; i < strips; i++) {
    const [r, g, b] = turboAt(i / (strips - 1));
    doc.setFillColor(Math.round(r), Math.round(g), Math.round(b));
    doc.rect(x + i * sw, barY, sw + 0.5, barH, "F");
  }
  doc.setDrawColor(...LINE);
  doc.rect(x, barY, w, barH);
  doc.setFont("helvetica", "normal");
  doc.setFontSize(7.5);
  doc.setTextColor(...MUTE);
  doc.text("Low complexity (smooth / normal)", x, barY + barH + 9);
  doc.text("High complexity (rough / suspicious)", x + w, barY + barH + 9, { align: "right" });
}

/** Draw the multifractal f(alpha) spectrum as a crisp vector plot (no image raster). */
function drawSpectrumChart(doc: jsPDF, result: AnalysisResponse, x: number, y: number, w: number, h: number): void {
  doc.setTextColor(...INK);
  doc.setFont("helvetica", "bold");
  doc.setFontSize(12);
  doc.text("Multifractal spectrum  f(alpha)", x, y);
  doc.setFont("helvetica", "normal");
  doc.setFontSize(8.5);
  doc.setTextColor(...MUTE);
  doc.text(`Spectrum width (alpha spread) = ${result.fractal.multifractalWidth.toFixed(3)}`, x + w, y, { align: "right" });

  const plotX = x + 34;
  const plotY = y + 16;
  const plotW = w - 44;
  const plotH = h - 42;

  const pts = result.fractal.spectrum
    .filter((p) => Number.isFinite(p.alpha) && Number.isFinite(p.falpha))
    .map((p) => ({ a: p.alpha, f: p.falpha }))
    .sort((p, q) => p.a - q.a);

  // Axes frame.
  doc.setDrawColor(...LINE);
  doc.setLineWidth(0.6);
  doc.rect(plotX, plotY, plotW, plotH);

  if (pts.length < 3) {
    doc.setTextColor(...MUTE);
    doc.setFontSize(9);
    doc.text("Spectrum unavailable for this image.", plotX + 10, plotY + plotH / 2);
    axisLabels(doc, plotX, plotY, plotW, plotH);
    return;
  }

  const aMin = Math.min(...pts.map((p) => p.a));
  const aMax = Math.max(...pts.map((p) => p.a));
  const fMin = Math.min(...pts.map((p) => p.f));
  const fMax = Math.max(...pts.map((p) => p.f));
  const sx = (a: number) => plotX + ((a - aMin) / (aMax - aMin || 1)) * plotW;
  const sy = (f: number) => plotY + plotH - ((f - fMin) / (fMax - fMin || 1)) * plotH;

  // Gridlines (quarters).
  doc.setDrawColor(238, 241, 246);
  doc.setLineWidth(0.4);
  for (let i = 1; i < 4; i++) {
    const gy = plotY + (plotH * i) / 4;
    doc.line(plotX, gy, plotX + plotW, gy);
    const gx = plotX + (plotW * i) / 4;
    doc.line(gx, plotY, gx, plotY + plotH);
  }

  // Spectrum curve.
  doc.setDrawColor(...CYAN);
  doc.setLineWidth(1.4);
  for (let i = 1; i < pts.length; i++) {
    doc.line(sx(pts[i - 1].a), sy(pts[i - 1].f), sx(pts[i].a), sy(pts[i].f));
  }
  // Data points.
  doc.setFillColor(...BLUE);
  for (const p of pts) doc.circle(sx(p.a), sy(p.f), 1.6, "F");

  // Axis tick values.
  doc.setTextColor(...MUTE);
  doc.setFontSize(7.5);
  doc.text(aMin.toFixed(2), plotX, plotY + plotH + 11);
  doc.text(aMax.toFixed(2), plotX + plotW, plotY + plotH + 11, { align: "right" });
  doc.text(fMax.toFixed(2), plotX - 4, plotY + 4, { align: "right" });
  doc.text(fMin.toFixed(2), plotX - 4, plotY + plotH, { align: "right" });

  axisLabels(doc, plotX, plotY, plotW, plotH);
}

function axisLabels(doc: jsPDF, plotX: number, plotY: number, plotW: number, plotH: number): void {
  doc.setTextColor(...MUTE);
  doc.setFontSize(8);
  doc.text("alpha (singularity strength)", plotX + plotW / 2, plotY + plotH + 24, { align: "center" });
  doc.text("f(alpha)", plotX - 22, plotY + plotH / 2, { align: "center", angle: 90 });
}

function sectionTitle(doc: jsPDF, title: string, x: number, y: number): number {
  const W = doc.internal.pageSize.getWidth();
  doc.setTextColor(...INK);
  doc.setFont("helvetica", "bold");
  doc.setFontSize(12);
  doc.text(title, x, y);
  doc.setDrawColor(...LINE);
  doc.line(x, y + 8, W - 42, y + 8);
  return y + 28;
}
