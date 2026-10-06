# AI-Based Detection of Abnormalities in X-Ray Images Using Fractal Dimension Analysis

> BE Final-Year Project — implementation plan & winning strategy
> Stack: **React + TypeScript + Framer Motion** (frontend) · **FastAPI + PyTorch** (backend)

---

## Context

This is a from-scratch BE capstone. The goal is not just "a model that classifies X-rays" — it is to make **Fractal Dimension Analysis** the star of the show (it's in the title, so judges will grade on it) while still hitting strong accuracy on a hard, credible problem.

**Decisions locked in (from clarification):**
- **Domain:** Multi-disease chest X-ray (NIH ChestX-ray14 label space, 14 pathologies).
- **Model:** Hybrid — a pretrained CNN fused with hand-computed fractal features.
- **Deliverable focus:** A polished, deployable **web app** (React + FastAPI).
- **Compute:** Google Colab for the small training step.

**The key insight that makes this feasible & impressive:** we use **[TorchXRayVision](https://github.com/mlmed/torchxrayvision)** — a maintained library shipping a DenseNet-121 pretrained on NIH + CheXpert + MIMIC + PadChest that already outputs ~18 pathology probabilities and a 1024-dim embedding. We do **not** train a 14-class network from scratch. We only train a **small fusion head** (CNN embedding + fractal feature vector → refined predictions), which fits comfortably in one Colab session. If training time runs out, the app still works on pretrained-only inference — graceful degradation.

**Intended outcome:** An upload-an-X-ray → get-a-diagnosis-with-explanations web app whose signature feature is a **local fractal-dimension heatmap** that visually shows *where* texture complexity is abnormal — directly tying the fractal theme to model explainability.

---

## Why fractal dimension for X-rays (the thesis, in one paragraph)

Biological tissue textures (lung parenchyma, bone trabeculae, vasculature) are statistically **self-similar** — they look similar across scales. **Fractal Dimension (FD)** quantifies that texture complexity as a single number; disease changes texture (e.g. consolidation, fibrosis, effusion) and therefore changes FD. But FD alone is coarse — two very different textures can share an FD. So we add **lacunarity** (how "gappy"/heterogeneous the texture is) and the **multifractal spectrum f(α)** (the *range* of local scaling behaviours). Together these form a compact, interpretable feature vector that complements what a CNN sees. That complementarity is the novelty we sell.

---

## What will make this WIN (differentiators — build these deliberately)

1. **Local Fractal Dimension heatmap** — slide a window across the image, compute FD per patch, render as a color overlay. This is the signature visual: it *shows* the fractal analysis, and doubles as explainability. Nobody else will have this.
2. **Multifractal spectrum + lacunarity**, not just a single FD number — signals real depth beyond "I computed box-counting."
3. **Validation on mathematical fractals** — unit-test the FD code on a Sierpinski triangle (FD ≈ 1.585), Koch curve (≈ 1.26), Cantor set. Proving your measure is *correct* against known ground truth is a killer slide in the viva and a table in the report.
4. **Dual explainability** — show CNN **Grad-CAM** attention *next to* the fractal texture map. "Where the neural net looks" vs "where the texture is anomalous" is a genuinely interesting comparison.
5. **Real medical format support** — accept **DICOM** (via `pydicom`), not just PNG/JPG. Shows clinical seriousness.
6. **Interactive methodology page** — a live box-counting animation on the page so a judge instantly *gets* what fractal dimension means. Demonstrates you understand it, not just wrapped a library.
7. **Multi-disease** via SOTA pretrained weights = high accuracy for free, credible clinical scope.

---

## System Architecture

```
┌──────────────────────────────┐        ┌────────────────────────────────────────────┐
│  Frontend (React + TS + FM)   │  HTTP  │  Backend (FastAPI)                          │
│                               │ ─────► │                                              │
│  Upload → Scanning animation  │        │  1. Preprocess  (DICOM/PNG, CLAHE, resize)   │
│  Result: predictions +        │ ◄───── │  2. Segment lungs (optional U-Net)           │
│  heatmaps + fractal charts    │  JSON  │  3a. CNN branch  → probs + 1024-d embedding  │
│                               │        │  3b. Fractal branch → FD, lacunarity, f(α),  │
└──────────────────────────────┘        │       local-FD map                           │
                                         │  4. Fusion head → refined predictions        │
                                         │  5. Explain: Grad-CAM + local-FD heatmap     │
                                         │  6. Return JSON (+ base64 heatmap images)    │
                                         └────────────────────────────────────────────┘
```

---

## Phase 0 — Model preparation (Google Colab, one-time)

Even though the deliverable is the web app, the hybrid model must exist. Keep this minimal.

1. `pip install torchxrayvision` — load `xrv.models.DenseNet(weights="densenet121-res224-all")`.
2. Take a subset of NIH ChestX-ray14 (public on Kaggle). For each image, extract: (a) the DenseNet **1024-d embedding**, (b) the **fractal feature vector** (global FD, DBC, lacunarity at several box sizes, multifractal width Δα, a few f(α) points).
3. Train a **small fusion MLP** (`[1024 + ~15] → 256 → num_labels`, sigmoid, BCE loss) — minutes on Colab GPU.
4. Evaluate: per-pathology **AUC**, and compare **CNN-only vs CNN+fractal fusion** (this delta is your headline result — even a small AUC gain proves the fractal contribution).
5. Export `fusion_head.pt` (a few MB) → commit to `backend/weights/`.

**Fallback if time-limited:** ship pretrained-only inference; the fractal features still render in the UI. The app is never blocked on training.

**Deliverable:** `notebooks/01_fractal_features.ipynb`, `notebooks/02_train_fusion.ipynb`, `backend/weights/fusion_head.pt`, a results table (CNN-only vs fusion AUC).

---

## Phase 1 — Backend: fractal core (do this first, it's the thesis)

`backend/app/services/fractal.py` — pure NumPy/SciPy, no ML. Implement and **unit-test**:

- **Box-counting FD** — threshold/edge-map → count non-empty boxes across scales → slope of `log N` vs `log 1/s`.
- **Differential Box Counting (DBC)** (Sarkar–Chaudhuri) — operates on the grayscale intensity surface directly (better for medical texture than binary box-counting).
- **Lacunarity** — gliding-box algorithm, `Λ(r) = E[M²]/E[M]²`.
- **Multifractal spectrum** — moments `q`, compute `τ(q)`, Legendre transform → `α`, `f(α)`; report spectrum width `Δα`.
- **Local FD map** — slide a 64×64 window (stride 32), DBC per window → 2-D array → upscale + colormap. **This is the signature feature.**

`backend/tests/test_fractal.py` — assert FD of generated **Sierpinski (≈1.585)**, **Koch (≈1.26)** within tolerance. This proves correctness.

---

## Phase 2 — Backend: CNN + fusion + explainability

- `services/preprocessing.py` — load PNG/JPG **and DICOM** (`pydicom`), grayscale, **CLAHE** contrast enhancement (`cv2.createCLAHE`), resize to 224, normalize to TorchXRayVision's expected range.
- `services/segmentation.py` — *optional* lung-field U-Net (TorchXRayVision ships a segmentation model) to focus fractal analysis on the lungs. Build after core works; degrade gracefully if skipped.
- `services/cnn_model.py` — load TorchXRayVision DenseNet once at startup; return per-pathology probabilities + 1024-d embedding.
- `services/gradcam.py` — Grad-CAM on the DenseNet's last conv block → attention overlay.
- `services/fusion.py` — load `fusion_head.pt`, concat embedding + fractal vector → refined predictions.
- `services/pipeline.py` — orchestrates: preprocess → (segment) → CNN + fractal in parallel → fuse → explain → assemble response.
- `utils/images.py` — encode heatmaps to base64 PNG, apply colormaps (`cv2.applyColorMap`).

---

## Phase 3 — Backend: FastAPI surface

```
backend/
  app/
    main.py                 # app, CORS, router registration, model warm-load on startup
    api/routes_analyze.py   # POST /api/analyze  (multipart image → full result)
    api/routes_health.py    # GET  /api/health
    core/config.py          # pydantic-settings
    models/schemas.py       # Pydantic request/response (mirror in frontend TS types)
    services/…              # phases 1–2 above
    tests/                  # pytest
  weights/                  # fusion_head.pt (+ cached model weights)
  requirements.txt
  Dockerfile
```

**`POST /api/analyze` response shape:**
```jsonc
{
  "predictions": [ { "label": "Pneumonia", "probability": 0.87 }, ... ],  // sorted desc
  "topFinding": { "label": "...", "probability": 0.87, "severity": "high" },
  "fractal": {
    "boxCountingFD": 1.83, "dbcFD": 2.41, "lacunarity": 1.12,
    "multifractalWidth": 0.34,
    "spectrum": [ { "alpha": 0.9, "falpha": 0.4 }, ... ]   // for the f(α) chart
  },
  "overlays": {
    "original":     "data:image/png;base64,…",
    "localFdMap":   "data:image/png;base64,…",
    "gradcam":      "data:image/png;base64,…"
  },
  "meta": { "model": "densenet121-res224-all + fusion", "processingMs": 1240 }
}
```

**Libs:** `fastapi, uvicorn[standard], python-multipart, torch, torchvision, torchxrayvision, scikit-image, opencv-python-headless, numpy, scipy, pydicom, pillow, pydantic-settings, pytest`.

---

## Phase 4 — Frontend: scaffold & integration (React + TS + Vite)

```
frontend/
  src/
    main.tsx · App.tsx · router
    routes/         Home.tsx · Analyze.tsx · Methodology.tsx
    components/
      UploadDropzone.tsx      # animated drag-drop (Framer Motion)
      LoadingScanner.tsx      # X-ray "scanning line" animation during processing
      AnalysisView.tsx        # orchestrates the result layout
      HeatmapViewer.tsx       # original + toggle FD-map / Grad-CAM, opacity slider
      PredictionList.tsx      # staggered animated probability bars
      ConfidenceGauge.tsx     # animated radial gauge for top finding
      FractalMetricsPanel.tsx # FD / lacunarity cards with count-up animation
      MultifractalChart.tsx   # f(α) spectrum (Recharts)
      BoxCountingDemo.tsx     # interactive live demo on Methodology page
    lib/api.ts                # typed fetch client
    types/analysis.ts         # TS types mirroring Pydantic schemas
    theme/                    # dark clinical design tokens
```

**Build order:** typed API client + types → upload → wire `/api/analyze` → render raw result → then layer premium UI.

---

## Phase 5 — Premium UI & animations (Framer Motion) — the "wow"

- **Dark, clinical aesthetic**: deep slate background, glass cards, a single medical-cyan accent, subtle grid. Reads as a real diagnostic tool.
- **Scanning animation**: a horizontal scan line sweeps the uploaded X-ray while the backend works — instantly on-theme, hides latency.
- **Staggered reveal**: prediction bars animate in with stagger; FD numbers **count up**; gauge sweeps to value (`AnimatePresence` + `motion`).
- **Heatmap blend**: opacity slider dissolves the local-FD map / Grad-CAM over the original; tabs switch "CNN attention ⇄ Fractal texture."
- **Methodology page**: live box-counting demo (grid subdivides, counts boxes, plots the log-log line and slope = FD) — makes the concept click for judges.
- **Report export**: "Download PDF report" (findings + metrics + heatmaps) — a professional touch for the evaluation.

---

## Phase 6 — Deployment (optional but strong)

- `Dockerfile` for backend + `docker-compose.yml` (backend + frontend).
- Frontend → Vercel/Netlify; backend → Render/Railway/HF Spaces (CPU inference is fine; keep model warm-loaded at startup).

---

## Milestone timeline (suggested)

| # | Milestone | Output |
|---|-----------|--------|
| 1 | Repo scaffold, envs, TorchXRayVision loads & runs on a sample | Both apps boot |
| 2 | Fractal core + fractal unit tests passing | `fractal.py`, green tests |
| 3 | Colab: fractal features + fusion head trained & exported | `fusion_head.pt`, AUC table |
| 4 | FastAPI `/api/analyze` end-to-end (JSON + overlays) | Working API |
| 5 | Frontend upload → results rendered (unstyled) | Functional app |
| 6 | Premium UI, animations, heatmap viewer, charts | Demo-ready app |
| 7 | Methodology page, PDF report, deploy | Polished, hosted |
| 8 | Report + slides + demo script | Submission pack |

---

## Verification & testing

- **Fractal correctness (headline):** `pytest backend/tests/test_fractal.py` — Sierpinski FD ≈ 1.585, Koch ≈ 1.26 within tolerance.
- **Model contribution:** Colab notebook prints CNN-only vs CNN+fractal **per-pathology AUC** — the delta justifies the fractal branch.
- **API smoke test:** `POST /api/analyze` with a sample chest X-ray → inspect JSON (predictions sorted, all three overlays non-empty, fractal metrics in plausible ranges).
- **End-to-end:** upload a known-pneumonia image in the UI → high pneumonia probability, heatmaps highlight lung fields, spectrum chart renders.
- **Sanity:** a clearly normal image → low probabilities, flatter/lower local-FD map.

---

## Risks & mitigations

| Risk | Mitigation |
|------|-----------|
| Colab training time runs out | Ship pretrained-only inference; fractal features still shown (graceful degradation) |
| CPU inference too slow | Warm-load model at startup; cache; 224² is light; show scanning animation to mask latency |
| Multi-label metrics look weak on a hard dataset | Lead the demo with strong pathologies (Pneumonia, Effusion, Cardiomegaly); report AUC per-class, not accuracy |
| Segmentation adds complexity | It's optional — full pipeline works without it; add last |
| DICOM edge cases | Support PNG/JPG first; add `pydicom` path as an enhancement |

---

## Open items to confirm before/if building
- Whether to include lung **segmentation** in v1 (recommended: add after core pipeline works).
- Whether the **PDF report export** is in-scope for v1 or a stretch goal.
- Deployment target (local demo only vs hosted).
