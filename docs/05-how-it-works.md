# 5. How It Works — Follow One X-Ray Through The App

Let's trace exactly what happens from the moment you drop in an X-ray to the moment you see
results. This is the story to tell in your viva.

## The big picture

```
   YOU                FRONTEND (website)              BACKEND (brain)
   │                       │                              │
   │  drag X-ray  ───────► │                              │
   │                       │  send image ───────────────► │
   │                       │                              │  1. clean image
   │                       │                              │  2. check it's an X-ray
   │                       │                              │  3. find lungs
   │                       │                              │  4. fractal maths
   │                       │                              │  5. run AI model
   │                       │                              │  6. combine + draw heatmaps
   │                       │  ◄─────────── results (JSON) │
   │  ◄─ pretty results ── │                              │
```

Now the same thing, step by step, with the file that does each part.

---

### Step 1 — You upload an image
**File:** `frontend/src/components/UploadDropzone.tsx`
You drag a chest X-ray onto the box. The website grabs the file.

### Step 2 — The website sends it to the backend
**File:** `frontend/src/lib/api.ts`
It packages the image and sends it over the internet (locally) to the backend's "analyse"
door at `http://localhost:8000/api/analyze`. Meanwhile the scanning animation plays
(`LoadingScanner.tsx`).

### Step 3 — The backend receives it
**File:** `backend/app/api/routes_analyze.py`
This checks the file is a valid image and not too big, then hands it to the conductor.

### Step 4 — The conductor runs everything in order
**File:** `backend/app/services/pipeline.py`
This is the "manager" that calls each specialist in turn:

1. **Prepare the image** — `preprocessing.py`
   Turns it black-and-white, makes the texture clearer, resizes it. Also reads medical
   **DICOM** files, not just PNG/JPG.

2. **Check it's really an X-ray** — `validation.py` 🚫
   Is it grayscale like an X-ray? Can it find lungs? If not (a selfie, a screenshot, text),
   it stops here and politely refuses — no fake results.

3. **Find the lungs** — `segmentation.py`
   A small AI outlines the lungs so we study only the lungs, not the ribs/heart/edges.
   (If this AI isn't installed, it just uses the whole image — no crash.)

4. **Do the fractal maths** — `fractal.py`  ⭐ (your title!)
   Measures the texture of the lungs:
   - **Fractal dimension** (roughness),
   - **Lacunarity** (gappiness),
   - **Multifractal spectrum** (the roughness curve),
   - **Local FD map** (roughness of every little patch → becomes the heatmap).

5. **Run the disease AI** — `cnn_model.py`
   The ready-made TorchXRayVision model looks at the image and gives a % chance for 14+
   diseases. It also gives a "fingerprint" of the image.

6. **Make the AI heatmap** — `gradcam.py`
   Draws where the AI was looking.

7. **Combine both opinions** — `fusion.py`
   If you trained a fusion model, it mixes the fractal numbers + AI fingerprint into a
   sharper answer. (If not, it just uses the AI's answer — no crash.)

8. **Turn numbers into pictures** — `utils/images.py`
   Colours the heatmaps (blue = smooth, red = rough) and packs them to send to the website.

### Step 5 — The backend sends results back
**File:** `backend/app/models/schemas.py` defines the exact shape of the reply. It's a
bundle (called **JSON**) containing: the disease list, the fractal numbers, the spectrum
points, and the heatmap images.

### Step 6 — The website displays everything beautifully
**File:** `frontend/src/components/AnalysisView.tsx` arranges it all:
- `ConfidenceGauge.tsx` → the big dial
- `PredictionList.tsx` → the disease % bars
- `HeatmapViewer.tsx` → the X-ray with toggleable heatmaps + fade slider
- `FractalMetricsPanel.tsx` → the fractal number cards
- `MultifractalChart.tsx` → the spectrum curve

### Step 7 — (Optional) You download the PDF
**File:** `frontend/src/lib/report.ts`
Builds a clean 2-page PDF report with everything, right in your browser.

---

## Why is it split into "frontend" and "backend"?
- The **frontend** is good at looking nice and being clicked (it runs in your browser).
- The **backend** is good at heavy maths and AI (it runs Python on your computer).
- They talk to each other over a simple web connection. This is how almost all modern apps
  are built — it's a professional design, and worth saying so in your viva.

## The "graceful fallback" trick (a good thing to mention)
The app is built so that if the AI parts aren't installed, it **still works** and just shows
the fractal analysis instead of crashing. That's why you could even run a quick version
without the heavy AI download. Good engineering = it never fully breaks.

👉 **Next:** [`06-viva-cheatsheet.md`](06-viva-cheatsheet.md) — nail the exam questions.
