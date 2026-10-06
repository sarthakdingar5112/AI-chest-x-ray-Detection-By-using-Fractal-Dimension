# 3. How To Run It — Step By Step (Beginner Friendly)

We'll get the app running on your own Windows PC. Take it slowly. Copy-paste the commands.

Your project has **two halves** that run at the same time:
- **Backend** (the "brain" — does the fractal math + AI) → runs on `http://localhost:8000`
- **Frontend** (the "face" — the website you click) → runs on `http://localhost:5173`

You'll open **two terminal windows**, one for each.

---

## Step 0 — Install two things (one-time)

### a) Python
- Any **Python 3.11 or newer works — including 3.14** (which you already have installed).
  PyTorch now ships CPU wheels for Python 3.14 on Windows, and this app runs on CPU, so
  your installed version is fine. Get it from https://www.python.org/downloads/ if needed.
- ⚠️ If you install fresh, **tick the box "Add Python to PATH."**
- The only thing Python 3.14 lacks is **GPU (CUDA) PyTorch** — but you don't need a GPU to
  *run* the app, and any model training is done on Google Colab anyway.

### b) Node.js
- Go to https://nodejs.org and install the **LTS** version. (You already have this — v22.)

> **What's a "terminal"?** It's a window where you type commands. On Windows, press the
> **Start** button, type **PowerShell**, and open it. That's your terminal.

---

## Step 1 — Start the Backend (Terminal #1)

Open PowerShell and paste these lines **one at a time** (press Enter after each):

```powershell
cd "D:\Nikhil\12.04.2025\SpikeHq\BE-Project\backend"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

> `py -m venv .venv` uses your default Python (3.14 — that's fine). If you ever want a
> specific version instead, write e.g. `py -3.12 -m venv .venv`.

**What each line does:**
1. Go into the backend folder.
2. Make a private Python workspace (a "virtual environment").
3. Turn it on (you'll see `(.venv)` appear at the start of the line).
4. Install everything the backend needs (~5–10 min the first time — grab a chai ☕).
5. Start the backend.

✅ **You'll know it worked** when you see something like:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
```
**Leave this window open.** It must keep running.

> First time you analyse an image, it downloads the AI model (~100 MB, ~1 min, once). Normal.

---

## Step 2 — Start the Frontend (Terminal #2)

Open a **second** PowerShell window (keep the first one running!) and paste:

```powershell
cd "D:\Nikhil\12.04.2025\SpikeHq\BE-Project\frontend"
npm install
npm run dev
```

**What each line does:**
1. Go into the frontend folder.
2. Download the website's building blocks (~2 min the first time).
3. Start the website.

✅ **You'll know it worked** when you see:
```
  ➜  Local:   http://localhost:5173/
```

---

## Step 3 — Use it! 🎉

1. Open your web browser and go to **http://localhost:5173**
2. Click **"Analyze"** in the top menu.
3. Drag a chest X-ray image onto the box (or click to browse).
   - Need an X-ray to test? See **"Where to get test X-rays"** below.
4. Watch the scanning animation, then see your results.
5. Click **"⤓ Download PDF report"** to save the report.

---

## Where to get test X-rays
Download a few chest X-ray images (PNG or JPG) from any of these free sources:
- Kaggle: search **"Chest X-Ray Images (Pneumonia)"** — has clear normal vs pneumonia images.
- Kaggle: **"NIH Chest X-rays sample"** (`nih-chest-xrays/sample`).
- Wikipedia — search "chest radiograph", right-click → save an image.

Save them somewhere easy like your Desktop, then drag them into the app.

---

## Do I need to "train" anything?
**No, not to run it.** The AI is already trained and downloads itself. Training is an
*optional bonus* that adds a fractal-fusion model and gives you accuracy numbers for your
report — see [`RUNBOOK.md`](../RUNBOOK.md) → "Level C" for that. For your demo, Steps 1–3
above are enough.

---

## Stopping the app
Click on each terminal window and press **Ctrl + C**. To run it again later, just repeat
Step 1 and Step 2 (you can skip the `pip install` / `npm install` lines after the first time
— but you DO need the `Activate.ps1` line for the backend each time).

---

## 🆘 Troubleshooting

| Problem | Fix |
|--------|-----|
| `py` is not recognized | Python isn't installed or not on PATH. Reinstall Python and tick "Add to PATH." |
| `Activate.ps1 cannot be loaded ... running scripts is disabled` | Run this once in PowerShell: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` then press `Y`. Try again. |
| `pip install` fails on `torch` | Rare on 3.14 now (CPU wheels exist). Make sure pip is current: `python -m pip install --upgrade pip`. If it still fails, use `py -3.12 -m venv .venv` as a fallback. |
| Website loads but says "ML stack unavailable" | The backend didn't install torch, OR you only did a partial install. Redo Step 1's `pip install -r requirements.txt`. |
| "Analyze" gives a network error | The backend (Terminal #1) isn't running, or you closed it. Restart Step 1. |
| Predictions are empty but fractal numbers show | Same as above — that's the app working in "fractal-only" mode without the AI. Fine for a quick test. |
| Port already in use | Something else is using 8000/5173. Close other apps or restart your PC. |

👉 **Next:** [`04-project-structure.md`](04-project-structure.md) — what every file does.
