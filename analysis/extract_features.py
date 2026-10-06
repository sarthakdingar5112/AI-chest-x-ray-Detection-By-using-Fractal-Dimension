"""
STEP 1 of the dataset study: extract fractal features from every image of a labelled dataset.

Usage
-----
    python analysis/extract_features.py \
        --data "path/to/X-Ray Image DataSet" \
        --out  analysis/results

`--data` must contain one sub-folder per class (e.g. No_findings/, Pneumonia/, Covid-19/).
Writes:
    results/features.csv        one row per image (class + all fractal features)
    results/local_fd_maps.npz   the 15x15 local-FD map of every image (for the mean maps)
    results/curves.npz          log-log box-counting curves (for the 'how FD is measured' figure)

Pre-processing is the same as the web app (grayscale -> CLAHE -> 512 px), then the fractal
features are measured on the central thorax region (rows/cols 64..448 of the 512 px image),
which approximates the lung fields when no segmentation model is available.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "backend"))
from app.services import fractal_fast as ff  # noqa: E402

SIZE = 512
ROI = (64, 448)  # central thorax crop used for global features
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Friendly names for the dataset's folders
CLASS_NAMES = {"No_findings": "Normal", "Pneumonia": "Pneumonia", "Covid-19": "COVID-19",
               "NORMAL": "Normal", "COVID19": "COVID-19"}


def load_gray(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    g = np.asarray(img, dtype=np.float64)
    g -= g.min()
    if g.max() > 0:
        g /= g.max()
    g8 = (g * 255).astype(np.uint8)
    g8 = CLAHE.apply(g8)
    g8 = cv2.resize(g8, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return g8.astype(np.float64) / 255.0


def process(item):
    path, label = item
    try:
        g = load_gray(path)
        fd_map = ff.local_fd_map(g, 64, 32)
        roi = g[ROI[0]:ROI[1], ROI[0]:ROI[1]]
        feats = ff.feature_dict(roi, fd_map=fd_map)
        feats["local_fd_mean"] = float(np.nanmean(fd_map))
        feats["local_fd_std"] = float(np.nanstd(fd_map))
        # simple intensity descriptors = the "is it just brightness?" control
        feats["intensity_mean"] = float(roi.mean())
        feats["intensity_std"] = float(roi.std())
        _, curve = ff.dbc_fd(roi)
        _, bcurve = ff.box_counting_fd(roi > roi.mean())
        return dict(file=os.path.basename(path), label=label, **feats), fd_map, curve, bcurve
    except Exception as e:  # keep going on a bad file
        print("FAILED", path, e, flush=True)
        return None


def collect(data_dir: Path, limit: int | None):
    items = []
    for sub in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        label = CLASS_NAMES.get(sub.name, sub.name)
        files = sorted(f for f in sub.iterdir() if f.suffix.lower() in {".png", ".jpg", ".jpeg"})
        if limit:
            files = files[:limit]
        items += [(str(f), label) for f in files]
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--limit", type=int, default=None, help="max images per class (quick tests)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    items = collect(Path(a.data), a.limit)
    print(f"{len(items)} images, {a.workers} workers", flush=True)

    t0 = time.time()
    rows, maps, dbc_curves, box_curves = [], [], [], []
    with Pool(a.workers) as pool:
        for k, res in enumerate(pool.imap(process, items, chunksize=4), 1):
            if res is None:
                continue
            r, m, c1, c2 = res
            rows.append(r); maps.append(m); dbc_curves.append(c1); box_curves.append(c2)
            if k % 50 == 0:
                print(f"  {k}/{len(items)}  {time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "features.csv", index=False)
    np.savez_compressed(out / "local_fd_maps.npz", maps=np.stack(maps), labels=df.label.values,
                        files=df.file.values)
    np.savez_compressed(out / "curves.npz", dbc=np.array([[n for _, n in c] for c in dbc_curves]),
                        dbc_sizes=np.array([s for s, _ in dbc_curves[0]]),
                        box=np.array([[n for _, n in c] for c in box_curves]),
                        box_sizes=np.array([s for s, _ in box_curves[0]]),
                        labels=df.label.values)
    print(df.groupby("label").size().to_string())
    print(f"done in {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
