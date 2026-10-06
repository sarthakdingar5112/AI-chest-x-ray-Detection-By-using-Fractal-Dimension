# RUNBOOK — get FractalScan fully working

Two ways to run it, from fastest to fullest:

| Level | What you get | Effort |
|-------|--------------|--------|
| **A. Fractal-only** | Full fractal analysis + heatmaps, **no disease predictions** | 5 min, no ML install |
| **B. CNN + fractal** | Everything + **14+ disease predictions** (pretrained, auto-downloaded) | ~15 min |
| **C. + Fusion head** | Everything + your **trained fusion model** and the CNN-vs-fusion AUC table for your report | + training time |

You do **not** need to train anything to get disease predictions — TorchXRayVision downloads a
pretrained model automatically. Training (Level C) only adds the fractal *fusion* and the
accuracy-comparison numbers for your thesis.

---

## 0. Prerequisites

- **Node.js 18+** (you have v22) — for the frontend.
- **Python 3.11+ — including 3.14** (which you already have). As of PyTorch 2.9+, **CPU
  wheels for Python 3.14 are published on Windows/macOS**, and this app runs on CPU, so your
  installed 3.14 works. The only gap is **GPU (CUDA) wheels for 3.14**, which you don't need
  to *run* the app (training is done on Colab).
  - Check what you have: `py -0p` (lists installed Python versions on Windows).
- ~2 GB disk for the Python ML packages; first CNN run downloads ~100 MB of model weights.

---

## Level A — Fractal-only (fastest, proves the app runs)

```powershell
# Backend
cd backend
py -m venv .venv                  # your default Python (3.14) is fine
.\.venv\Scripts\Activate.ps1
pip install fastapi "uvicorn[standard]" python-multipart pydantic-settings numpy scipy scikit-image pillow pydicom
uvicorn app.main:app --reload --port 8000
```
```powershell
# Frontend (new terminal)
cd frontend
npm install
npm run dev
```
Open **http://localhost:5173**, upload any chest X-ray image → you get the fractal metrics,
the local-FD heatmap, the multifractal spectrum, and the PDF report. The predictions panel
will say the ML model isn't loaded — that's expected at this level.

---

## Level B — Add disease predictions (recommended)

Same as A, but install the full backend requirements (this adds torch + torchxrayvision):

```powershell
cd backend
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

- On the **first** analysis request, TorchXRayVision downloads the pretrained DenseNet
  (`densenet121-res224-all`) and the lung-segmentation model. This takes ~30–60 s once, then
  it's cached in your user folder.
- After that: upload an X-ray → you get **14+ pathology probabilities**, **Grad-CAM**,
  **lung-field segmentation**, and the fractal analysis focused on the lungs.

> CPU is fine — inference on one 224×224 image is well under a second. No GPU required to *run*.

That's a fully working, demo-ready application. Levels A/B need **no dataset and no training.**

---

## Level C — Train the fusion head (for the "novelty" + report numbers)

This is the one model you train. It learns to combine the CNN's features with your fractal
features, and prints a **CNN-only vs fusion AUC** comparison — your headline result.

### C.1 Get the data

**NIH ChestX-ray14** on Kaggle:
- Full set: https://www.kaggle.com/datasets/nih-chest-xrays/data (~112k images, ~42 GB).
- **Sample set (use this):** https://www.kaggle.com/datasets/nih-chest-xrays/sample
  (~5,600 images, ~4 GB) — plenty to train the small fusion head.

Download gives you:
- an `images/` folder (PNG chest X-rays), and
- **`Data_Entry_2017.csv`** — the labels (the training script reads this format natively).

Easiest path is **Google Colab** (free GPU, and you can pull the dataset straight from Kaggle):

```python
# In a Colab notebook
!pip install kaggle
# upload your kaggle.json (Account → Create New API Token), then:
!mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d nih-chest-xrays/sample -p /content --unzip
```

### C.2 Run training

Upload `training/train_fusion.py` and `backend/app/services/fractal.py` to Colab (keep the
`backend/app/...` folder structure), then:

```bash
pip install torchxrayvision scikit-image scikit-learn pillow
python train_fusion.py \
  --images  /content/sample/images \
  --labels  /content/sample/sample_labels.csv \
  --out     fusion_head.pt \
  --limit   3000
```
(For the full dataset use `Data_Entry_2017.csv` and point `--images` at the parent folder —
the script walks all 12 `images_*` subfolders automatically.)

It prints per-pathology AUC like:
```
Effusion             0.881 -> 0.897  (+0.016)
Cardiomegaly         0.902 -> 0.908  (+0.006)
...
```
That CNN-only → fusion delta is exactly what you put in your report/viva.

### C.3 Install the trained model

Download `fusion_head.pt` from Colab and drop it into **`backend/weights/fusion_head.pt`**.
Restart the backend. Done — the model label in results now reads
`densenet121-res224-all + fusion`, and predictions are the fused output.

---

## Testing it works

- `curl http://localhost:8000/api/health` → `{"status":"ok","mlAvailable":true,...}` (Level B/C).
- Backend fractal validation: `cd backend && pytest -q` → 9 tests pass (Sierpinski ≈ 1.585 etc.).
- Where to get **test X-rays** to upload:
  - Any image from the NIH/Kaggle sample above.
  - Kaggle "Chest X-Ray Images (Pneumonia)" dataset (clear normal vs pneumonia examples).
  - Wikipedia/Radiopaedia chest X-ray images (PNG/JPG) for a quick manual demo.

---

## Deploying "for real" (optional)

- **Frontend** → Vercel/Netlify (`npm run build`, serve `dist/`).
- **Backend** → Render / Railway / Hugging Face Spaces (Docker). A `backend/Dockerfile` is
  included; it uses Python 3.11 so torch installs cleanly. Bake the model weights into the
  image or let them download on first boot.
- Set the frontend's `VITE_API_BASE` env var to your deployed backend URL.

---

## Honest scope note (say this in your viva)

This is a **decision-support / research** tool, not a certified diagnostic device. The CNN is
pretrained on public data and the fractal features add explainable texture evidence. Report
**AUC per pathology**, not a single "accuracy" number — that's the correct metric for
multi-label chest X-ray and shows you understand the problem.
