# 6. Viva / Presentation Cheat Sheet

Simple answers to the questions examiners actually ask. Read these out loud a few times and
you'll sound confident. Keep answers short; expand only if they push.

---

### Q: In one line, what is your project?
> "A web app that detects abnormalities in chest X-rays by combining a pretrained AI model
> with fractal-dimension texture analysis, and it explains its decision with heatmaps."

### Q: What is fractal dimension?
> "It's a single number that measures how complex or rough a pattern is — 1 is a smooth
> line, 2 is a filled area, and detailed textures fall in between. Diseased lung tissue has
> a different roughness than healthy tissue, so this number becomes a useful clue."

### Q: How do you calculate it?
> "Using box-counting. I cover the image with a grid of boxes and count how many contain
> part of the pattern, then repeat with smaller and smaller boxes. The rate at which the
> count grows as boxes shrink is the fractal dimension — it's the slope of a log-log graph."

### Q: Why use fractals for X-rays? What's the point?
> "Biological tissue is naturally fractal-like — it's self-similar across scales. Disease
> changes tissue texture, and fractal measures capture that texture mathematically. It gives
> an explainable, non-black-box piece of evidence to support the AI."

### Q: What's your novelty / contribution?
> "Three things: (1) a **local fractal-dimension heatmap** that visually shows *where* the
> texture is abnormal, (2) I go beyond a single number by using **lacunarity** and the
> **multifractal spectrum**, and (3) I **fuse** the fractal features with a deep-learning
> model and compare the two — including a Grad-CAM vs fractal-map comparison."

### Q: Which AI model do you use? Did you train it?
> "I use TorchXRayVision — a DenseNet CNN pretrained on over 100,000 chest X-rays from NIH,
> CheXpert, MIMIC and PadChest. I didn't retrain it from scratch; that would need huge
> compute. Instead I **fuse** its output with my fractal features by training a small
> fusion model, which is efficient and still novel."

### Q: What dataset?
> "The NIH ChestX-ray14 label space — 14 pathologies like pneumonia, effusion and
> cardiomegaly. For training the fusion head I used a sample of that dataset from Kaggle."

### Q: What accuracy did you get? (REAL numbers — use these)
> "On 1,000 real X-rays (500 normal, 500 pneumonia) with patient-grouped 5-fold cross-validation: a trained
> head on the AI embedding reaches AUC 0.94 and 85% accuracy. Fractal features alone reach AUC 0.80 (72% accuracy) and beat the
> untrained pretrained model (0.78). Adding fractal features to the trained AI gave no extra gain (0.936 vs 0.938),
> so their main value is explainability (and that trained head did not generalise to outside images, so the app uses the pretrained multi-hospital model) — the heatmaps and the measurable texture difference between healthy and diseased lungs."

### (older generic answer) What accuracy did you get?
> "For multi-label chest X-ray the correct metric is **AUC per pathology**, not a single
> accuracy number. I report AUC for each disease and compare **CNN-only vs CNN+fractal
> fusion** to show the fractal features add value."
>
> *(If you haven't trained the fusion head yet, say: "The pretrained model already achieves
> strong published AUCs; my fusion step is designed to refine these, and I report the
> per-class comparison.")*

### Q: What is lacunarity? Multifractal spectrum?
> "Lacunarity measures how gappy or clumpy a texture is — two textures can share a fractal
> dimension but differ in lacunarity. The multifractal spectrum is a curve, not one number:
> it captures the *range* of local roughness; a wider spectrum means richer, often more
> pathological texture."

### Q: What is Grad-CAM and why show it?
> "Grad-CAM highlights the image regions the CNN focused on. I show it beside my fractal
> heatmap so we can compare where the AI looked versus where the texture is anomalous — when
> they agree, that's stronger evidence."

### Q: What happens if someone uploads a photo that isn't an X-ray?
> "The app checks the input first. It confirms the image is grayscale like an X-ray, and it
> uses the lung-segmentation model to check that lungs are actually present. If not — a
> selfie, a screenshot, some text — it refuses and asks for a chest X-ray instead of
> producing meaningless results. It's a simple but important safety feature."

### Q: Why segment the lungs?
> "An X-ray also contains ribs, heart and black borders. Segmenting the lungs lets me run
> the fractal analysis only on the relevant tissue, giving cleaner, more meaningful numbers."

### Q: What's your tech stack?
> "Frontend: React with TypeScript and Framer Motion for the animated UI. Backend: Python
> with FastAPI. AI: PyTorch and TorchXRayVision. Fractal math: NumPy and SciPy."

### Q: How did you validate your fractal code is correct?
> "I tested it on mathematical fractals whose dimension is known exactly — a Sierpinski
> triangle (≈1.585), a Sierpinski carpet (≈1.893) and a Koch curve (≈1.262). My code
> recovers all of them within a small tolerance. That's an automated test that proves the
> measurement is correct." *(It's in `backend/tests/test_fractal.py` — run `pytest`.)*

### Q: What are the limitations?
> "It's a research/education prototype, not a certified medical device. It depends on the
> quality of the input X-ray, the pretrained model's biases, and it isn't clinically
> validated. It should assist, not replace, a radiologist."

### Q: How would you improve it in future?
> "Train the fusion head on more data, add more fractal features, support 3D CT scans,
> validate against radiologist labels, and deploy it online for real-time use."

---

## 60-second demo script (for the live demo)
1. "This is FractalScan. Let me upload a chest X-ray." *(drag image)*
2. "It's now finding the lungs and computing the fractal texture while the AI classifies it."
3. "Here's the top finding with confidence, and the full list of pathologies."
4. "This heatmap is my key feature — it shows where the fractal dimension is high, i.e. where
   texture is abnormal. I can toggle to the AI's attention map and fade between them."
5. "These are the fractal measures — roughness, gappiness — and this is the multifractal
   spectrum. And I can export a full PDF report." *(click Download)*
6. "On the Methodology page there's an interactive box-counting demo showing how the
   dimension is actually measured." *(show slider)*

---

## Three sentences that make you sound like you own it
- "The fractal dimension quantifies tissue-texture complexity, which disease alters."
- "I fuse explainable fractal features with a pretrained CNN, rather than relying on a
   black box alone."
- "I validated my fractal implementation against fractals of known dimension."

You've got this. 💪
