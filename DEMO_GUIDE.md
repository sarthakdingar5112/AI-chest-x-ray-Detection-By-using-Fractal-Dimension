# DEMO GUIDE — FractalScan (final-year project)

## Before the presentation (do this the night before, with internet)
1. Install **Python 3.11–3.13** and **Node.js 18+** if not installed.
2. Double-click **`1_SETUP_ONCE.bat`** (15–20 min).
3. Double-click **`2_START_BACKEND.bat`**. Wait until it says *CNN model loaded* (the first run downloads
   ~300 MB of AI weights — do this at home, not in the exam room).
4. Double-click **`3_START_FRONTEND.bat`** → browser opens at http://localhost:5173.
5. Test once with every picture in `test_images/`. Keep both black windows open during the demo.
> Mac/Linux: `cd backend && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && uvicorn app.main:app --port 8000` and in another terminal `cd frontend && npm install && npm run dev`.

## 5-minute demo script
1. **Home** (30 s) – "Our system detects chest X-ray abnormalities using AI plus fractal-dimension texture analysis."
2. **Methodology** (60 s) – drag the box-counting slider: "fractal dimension is the slope of this log-log line."
3. **Analyze** (90 s) – upload `04_tuberculosis.jpg`: point at the predictions, the lung outline, the **local-FD heatmap**
   (blue = smooth, red = rough), **Grad-CAM** ("where the AI looked"), the fractal numbers, then click **Download PDF report**.
   Then upload `NOT_xray_text.png` → it politely refuses (input validation).
4. **Dataset Study** (90 s) – *this answers the examiner*: "We analysed 1,000 real X-rays. Normal lungs average
   DBC fractal dimension 2.203; pneumonia 2.250. 9 of 10 fractal features differ significantly. Fractal features alone reach
   AUC 0.87; a brightness-only check only 0.73." Show the example images + the average-structure maps.
5. Close with limitations: research/education only, public data, not a medical device.

## If something fails on the day
- Backend shows "fractal-only" badge → the AI weights didn't load; the fractal analysis, heatmaps and PDF still work.
- Website can't reach backend → check the backend window is open on port 8000.
- Dataset Study tab works even without the backend (it only shows saved results).
