# FractalScan — AI Detection of Abnormalities in X-Ray Images via Fractal Dimension Analysis

> 🟢 **New here / not sure what any of this is?** Start with **[`docs/README.md`](docs/README.md)** —
> plain-English guides that explain what the project is, the theory, how to run it, the file
> structure, and viva questions. No prior knowledge needed.

A BE final-year project. Upload a chest X-ray → get multi-disease predictions **plus** an
explainable **fractal-dimension analysis** of the image texture: a local FD heatmap,
lacunarity, and the multifractal spectrum.

- **Frontend:** React + TypeScript + Vite + Framer Motion (premium clinical UI)
- **Backend:** FastAPI + PyTorch + TorchXRayVision (pretrained multi-label CNN) fused with
  hand-computed fractal features (NumPy/SciPy).

## Thesis in one line
Disease changes tissue texture; **fractal dimension**, **lacunarity**, and the
**multifractal spectrum** quantify that texture, complementing what a CNN sees — and the
**local FD heatmap** *shows* where the texture is abnormal.

## Repository layout
```
backend/    FastAPI app, fractal core, model pipeline, tests
frontend/   React + TS + Framer Motion web app
notebooks/  Colab notebooks: fractal features + fusion-head training
```

## Quick start

### Backend
```bash
cd backend
py -m venv .venv                 # Python 3.11+ works, including 3.14 (see note below)
.venv\Scripts\activate           # Windows PowerShell:  .venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
Open http://localhost:8000/docs for the interactive API.

> **Python version note:** the fractal core (`app/services/fractal.py`) runs on any Python
> 3.9+. The ML stack (`torch`, `torchvision`, `torchxrayvision`) now publishes **CPU wheels
> for Python 3.14** on Windows/macOS (PyTorch 2.9+), so a default 3.14 venv works for running
> the app on CPU. Only **GPU/CUDA** wheels for 3.14 are still unavailable — irrelevant here
> since inference runs on CPU and training is done on Colab. The app also degrades gracefully:
> if the ML deps are absent it still serves the fractal analysis.

### Frontend
```bash
cd frontend
npm install
npm run dev
```
Open http://localhost:5173.

## Tests
```bash
cd backend
pytest -q          # validates FD against known fractals (Sierpinski ≈ 1.585, etc.)
```

See `PLAN.md` for the full architecture and milestone plan.
