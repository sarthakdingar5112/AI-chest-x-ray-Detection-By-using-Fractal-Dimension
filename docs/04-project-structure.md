# 4. Project Structure — What Every File Does

Don't panic at the number of files. There are really just **4 areas**:

- `backend/` = the brain (Python: fractal math + AI)
- `frontend/` = the face (the website you click)
- `training/` = the optional "teach the fusion model" scripts
- `docs/` = these explanation files

Here's the full map with plain-English notes.

```
BE-Project/
│
├── README.md                  Short project intro
├── PLAN.md                    The full technical plan (advanced reading)
├── RUNBOOK.md                 Detailed run + training guide (advanced)
├── .gitignore                 Tells Git which files to ignore
│
├── docs/                      👈 THE SIMPLE GUIDES (you are here)
│   ├── README.md              Start-here index
│   ├── 01-what-is-this.md     What the project is
│   ├── 02-concepts-simple.md  The theory in simple words
│   ├── 03-how-to-run.md       How to run it
│   ├── 04-project-structure.md This file
│   ├── 05-how-it-works.md     The step-by-step flow
│   └── 06-viva-cheatsheet.md  Exam questions + answers
│
├── backend/                   🧠 THE BRAIN (Python + FastAPI)
│   ├── requirements.txt       List of Python packages to install
│   ├── Dockerfile             Recipe to run the backend on a server (deployment)
│   ├── pytest.ini             Settings for the tests
│   ├── weights/               Where the trained fusion model file goes (if you train one)
│   │
│   ├── app/                   The actual backend code
│   │   ├── main.py            🚪 The entry point — starts the web server
│   │   │
│   │   ├── api/               The "doors" the website talks to
│   │   │   ├── routes_analyze.py   Handles "here's an X-ray, analyse it"
│   │   │   └── routes_health.py    Handles "are you alive?" checks
│   │   │
│   │   ├── core/
│   │   │   └── config.py      Settings (allowed file size, addresses, etc.)
│   │   │
│   │   ├── models/
│   │   │   └── schemas.py     Defines the SHAPE of the data sent back (the "form")
│   │   │
│   │   ├── services/          ⭐ THE IMPORTANT LOGIC LIVES HERE
│   │   │   ├── fractal.py         ⭐ THE HEART: all the fractal math
│   │   │   ├── preprocessing.py   Cleans/prepares the uploaded image
│   │   │   ├── segmentation.py    Finds the lungs (AI)
│   │   │   ├── cnn_model.py       Runs the pretrained disease-detecting AI
│   │   │   ├── gradcam.py         Makes the "where the AI looked" heatmap
│   │   │   ├── fusion.py          Combines fractal + AI into a final answer
│   │   │   └── pipeline.py        🎯 THE CONDUCTOR: runs all the above in order
│   │   │
│   │   └── utils/
│   │       └── images.py      Turns math results into colored pictures
│   │
│   └── tests/                 Proof the fractal math is correct
│       ├── test_fractal.py         Checks FD against shapes with known answers
│       └── fractal_generators.py   Makes those test shapes (Sierpinski, Koch...)
│
├── frontend/                  🖥️ THE FACE (React + TypeScript website)
│   ├── package.json           List of website building blocks to install
│   ├── index.html             The base web page
│   ├── vite.config.ts         Build/settings for the website
│   │
│   └── src/                   The website code
│       ├── main.tsx           🚪 Starts the website
│       ├── App.tsx            The layout + top menu + page switching
│       │
│       ├── routes/            The 3 pages of the site
│       │   ├── Home.tsx           The landing/intro page
│       │   ├── Analyze.tsx        The upload-and-see-results page
│       │   └── Methodology.tsx    The "how fractals work" teaching page
│       │
│       ├── components/        Reusable pieces of the pages
│       │   ├── UploadDropzone.tsx      The drag-and-drop box
│       │   ├── LoadingScanner.tsx      The scanning animation
│       │   ├── AnalysisView.tsx        Arranges all the results
│       │   ├── HeatmapViewer.tsx       The X-ray + toggleable heatmaps
│       │   ├── PredictionList.tsx      The disease % bars
│       │   ├── ConfidenceGauge.tsx     The big confidence dial
│       │   ├── FractalMetricsPanel.tsx The fractal-number cards
│       │   ├── MultifractalChart.tsx   The spectrum curve chart
│       │   └── BoxCountingDemo.tsx     The interactive teaching toy
│       │
│       ├── lib/
│       │   ├── api.ts         Sends the image to the backend, gets results
│       │   └── report.ts      Builds the downloadable PDF report
│       │
│       ├── types/
│       │   └── analysis.ts    Describes the result data (matches the backend)
│       │
│       └── styles/
│           └── global.css     The colors, fonts, dark theme
│
└── training/                  🎓 OPTIONAL: teach the fusion model
    ├── train_fusion.py        Script that trains the fusion model (run on Google Colab)
    └── README.md              How to run the training
```

## If you only remember 3 files, remember these:
1. **`backend/app/services/fractal.py`** — the fractal math. This is literally your project
   title. If an examiner asks "show me your fractal code," open this.
2. **`backend/app/services/pipeline.py`** — the conductor that runs everything in order.
   Great for explaining the flow.
3. **`frontend/src/routes/Analyze.tsx`** — the main page users interact with.

## What's NOT in the tree (and that's fine)
- `node_modules/`, `.venv/`, `dist/`, `__pycache__/` — these appear after you install/run.
  They're auto-generated. Ignore them. (That's what `.gitignore` is for.)

👉 **Next:** [`05-how-it-works.md`](05-how-it-works.md) — follow one X-ray through the app.
