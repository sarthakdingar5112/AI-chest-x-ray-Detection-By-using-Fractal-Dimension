# 7. Dataset Study — "How does the structure look with and without disease?"

This answers the examiner's request directly. It is in the app (**Dataset Study** tab) and in
`analysis/` (code + results).

## What we did (say this in the viva)
1. Took a public labelled dataset: **500 normal ("No findings") + 500 pneumonia + 125 COVID-19**
   chest X-rays (GitHub: `muhammedtalo/COVID-19`, folder `X-Ray Image DataSet`).
   Normal and Pneumonia images come from NIH ChestX-ray14, same size (1024×1024), so they are comparable.
2. Ran the **same fractal pipeline as the web app** on every image
   (grayscale → CLAHE → 512 px → fractal features on the central thorax).
3. Compared the two groups statistically, drew average "structure maps", and tested whether
   fractal features alone can separate them.

## Headline results (Normal vs Pneumonia, n = 500 + 500)
| Finding | Number |
|---|---|
| Fractal features that differ significantly (Holm-corrected) | **9 of 10** |
| Grayscale surface FD (DBC): Normal → Pneumonia | **2.203 → 2.250 (+2.1 %)**, Cohen d = 1.38 (large) |
| Local FD mean | 2.102 → 2.122 (d = 0.89) |
| Lacunarity (r = 4) | 1.697 → 1.751 (+3.2 %) |
| Best single feature AUC (DBC FD) | **0.84** |
| Fractal-only classifier (Random Forest, patient-grouped 5-fold CV) | **AUC 0.865, accuracy 78.8 %, sens 76 %, spec 81.6 %** |
| Brightness-only control | AUC 0.728 |
| Fractal + brightness | AUC 0.893, accuracy 81.6 % |
| Map cells that differ significantly (FDR < 0.05) | 82 % — strongest in lower/central lung fields |

**Plain-language conclusion:** *Diseased (pneumonia) lungs have a more complex, rougher texture
(higher fractal dimension and lacunarity) than normal lungs, and the difference is spread over
the lung fields. Fractal features alone separate the two groups much better than a simple
brightness check — so fractal analysis carries real diagnostic information.*

One exception worth mentioning (shows honesty): **multifractal width did not differ** (p = 0.91),
and binary box-counting FD moved slightly *down* (1.861 → 1.854), while the grayscale (DBC) FD
moved *up*. Grayscale FD is the more informative measure here.

## Part 2 — Various diseases (what the examiner asked: "how does each disease look?")
Script: `analysis/multi_disease.py` (2,988 images, three public sources). Every disease is compared **only with normal
images from the same source** (scanner/compression differences can mimic disease effects).

| Disease (source, n disease / n normal) | DBC surface FD: normal → disease | Effect (Cohen d) | Significant? |
|---|---|---|---|
| Pneumonia (NIH-derived, 500 / 500) | 2.203 → 2.250 (+2.1 %) | +1.38 | **yes** |
| COVID-19 (Kaggle set, 712 / 800) | 2.320 → 2.286 (−1.5 %) | −0.63 | **yes** (opposite direction) |
| COVID-19 (Cohen collection, 342 / 15) | 2.263 → 2.280 | +0.27 | no (only 15 normals) |
| Fungal pneumonia (23 / 15) | 2.263 → 2.295 | +0.59 | no (too few) |
| Bacterial pneumonia (49 / 15) | 2.263 → 2.251 | −0.25 | no (too few) |
| Viral, non-COVID (36 / 15) | 2.263 → 2.259 | −0.09 | no (too few) |
| Tuberculosis (11 / 15) | 2.263 → 2.261 | −0.03 | no (too few) |

**How to explain it:** *different diseases leave different fractal signatures.* Pneumonia makes the lung texture
rougher and more complex (higher surface FD, higher lacunarity, higher local FD). COVID-19 (patchy ground-glass
opacities toward the periphery) shows the opposite surface-FD change and higher lacunarity, plus a lower multifractal
width. We do **not** claim a universal "diseased = higher FD" rule.

**What we cannot claim:** for TB, bacterial, viral and fungal pneumonia the public data has only 11–49 cases and 15
normals, so those results are exploratory (no significance). A larger labelled set (e.g. full NIH ChestX-ray14 with
14 diseases) would be the next step — the same scripts run on it unchanged.

Figures: `analysis/results/multi/fig6..fig9` (also on the Dataset Study tab). Re-run:
`python analysis/multi_disease.py --a <muhammedtalo/COVID-19> --b <education454/datasets> --c <ieee8023/covid-chestxray-dataset>`

**Q: How does the structure look for different diseases?** Show Fig. 9 (typical X-ray + texture map of each condition) and Fig. 6
(signature heat-map). Pneumonia: rougher/more complex. COVID-19: different signature (lower surface FD, higher lacunarity).
Small-sample diseases are reported as exploratory — we say so rather than over-claim.

## Fusion experiment: does fractal add value on top of the AI?
Trained on the same 1,000 images, patient-grouped 5-fold CV (682 patients), same code path as the web app,
strongly regularised MLP, hyper-parameters chosen by cross-validated log-loss, probabilities calibrated:

| Model | AUC | Accuracy |
|---|---|---|
| A. Pretrained AI, zero-shot "Pneumonia" score | 0.781 | – |
| Fractal features only (small MLP) | 0.797 | 72.1 % |
| B. AI embedding + trained head | **0.938** | 85.3 % |
| C. AI embedding + fractal features (fusion) | **0.936** | 85.5 % |
| Gain C − B | −0.002 (95 % CI −0.006 … +0.000) | no gain |

**Finding 1 – fractal adds no accuracy on top of the AI.** Fractal features alone (0.80) already beat the
*untrained* AI (0.78), so they carry real information, but the deep embedding already contains the same texture
information. The fractal contribution of this project is **explainability and an independent, interpretable
signal** (heatmaps, measurable numbers, group differences), not an accuracy boost.

**Finding 2 – a head trained on one dataset does not transfer.** The trained head looked excellent in
cross-validation, but when we served it in the web app on public chest X-rays from *other* sources, it called
healthy-looking images "pneumonia" (e.g. 96–100 %). Ablation showed the AI embedding (not the fractal inputs)
drives this: the head learned dataset-specific cues (hospital/scanner/positioning) — classic *dataset shift /
shortcut learning*; cross-validation inside one dataset cannot detect it. Therefore the head is kept only as an
experiment (`training/fusion_head_pneumonia_experiment.pt`) and the **web app uses the pretrained multi-label
model (trained on four large multi-hospital datasets) plus fractal analysis**. Proper external validation on
data from a different hospital is future work.

## Files
| File | What it is |
|---|---|
| `analysis/extract_features.py` | Step 1 – fractal features for every image → `features.csv` |
| `analysis/analyze.py` | Step 2 – statistics + all figures |
| `analysis/results/` | CSV tables + 5 figures + `summary.json` (already generated) |
| `training/train_fusion_pneumonia.py` | Trains + cross-validates the fusion head, exports `backend/weights/fusion_head.pt` |
| `backend/app/services/fractal_fast.py` | Vectorised fractal code (~50× faster); verified equal to the readable version by `tests/test_fractal_fast.py` |

## Re-run
```bash
pip install numpy scipy scikit-learn pandas matplotlib opencv-python-headless pillow pytest
git clone --depth 1 https://github.com/muhammedtalo/COVID-19
python analysis/extract_features.py --data "COVID-19/X-Ray Image DataSet"   # ~3 min
python analysis/analyze.py          --data "COVID-19/X-Ray Image DataSet"   # ~1 min
cd backend && pytest -q                                                    # 16 tests
```
Then copy `analysis/results/{fig*.png,summary.json}` and the two JSON tables into
`frontend/public/dataset-study/` if you re-generate.

## Honest limitations (say them before the examiner does)
- Image-level labels from a public dataset, not radiologist-verified for this study.
- Fractal features are texture statistics: differences are real **on average**, individual images overlap (AUC ≈ 0.87, not 1.0).
- Scanner/exposure can influence texture, so a **brightness-only control** is reported.
- Central-thorax crop is used instead of lung segmentation in the batch study (the web app uses segmentation when the model is installed).
- COVID-19 images come from other sources → shown for reference only, not in the statistics.
- Research/education tool, not a medical device.

## Likely examiner questions
**Q: How does a diseased structure look compared with a normal one?**
Diseased lungs show higher grayscale fractal dimension (≈ 2.25 vs 2.20) and higher lacunarity — more
complex, more heterogeneous texture (opacities, consolidation add irregular detail). Our average
local-FD maps show the increase concentrated over the lower and central lung fields (Fig. 2).

**Q: Did fractal features improve the AI's accuracy?** Honestly, no: AI-only head AUC 0.938 vs AI + fractal 0.936 (not significant).
Fractal features alone reach 0.80 and beat the untrained AI (0.78), so they are informative — their value is interpretability and an independent texture measurement.

**Q: Why isn't your trained model in the app?** We tested it on outside images and it over-called pneumonia (dataset shift / shortcut learning), so we kept the multi-hospital pretrained model and report the trained head as an experiment. We would rather show an honest limitation than a misleading demo.

**Q: Is it just brightness?** No. Brightness alone gives AUC 0.73; fractal features alone give 0.87; combined 0.89.

**Q: Why not 99 % accuracy?** Normal-vs-pneumonia from texture alone is hard and we use no deep network
here; we report honest cross-validated numbers grouped by patient so no patient appears in both train and test.

**Q: Which dataset / how many images?** 1,125 chest X-rays (500/500/125), public GitHub dataset derived from NIH ChestX-ray14 and Cohen's COVID collection.

**Q: How do you know your fractal code is right?** Unit tests recover Sierpinski triangle (1.585), carpet (1.893), Koch curve (1.262); the fast version is tested against the readable one.
