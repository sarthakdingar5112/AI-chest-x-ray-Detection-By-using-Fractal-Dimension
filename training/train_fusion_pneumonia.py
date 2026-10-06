"""
Train + validate the FUSION HEAD on the labelled Normal-vs-Pneumonia dataset.

What it does
------------
For every image it runs the *same code path as the web app*:
    preprocess (CLAHE, 512 px / 224 px)  ->  lung segmentation  ->  lung-ROI fractal vector
    DenseNet-121 (TorchXRayVision)       ->  1024-d embedding + zero-shot "Pneumonia" score
Features are cached to disk (so re-runs are instant), then three models are compared with
patient-grouped 5-fold cross-validation:

    A. CNN zero-shot   : the pretrained DenseNet's own "Pneumonia" output   (no training)
    B. CNN head        : small trained MLP on the 1024-d embedding only
    C. CNN + FRACTAL   : same MLP on [embedding || 11 fractal features]     <- our method

C vs B isolates the contribution of the fractal features (the project's headline claim).
Finally the fusion head (C) is trained on all data and exported to
training/fusion_head_pneumonia_experiment.pt. It is deliberately NOT placed in backend/weights/:
it does not generalise to chest X-rays from other sources (see docs/07-dataset-study.md).

Usage
-----
    python training/train_fusion_pneumonia.py --data "<...>/X-Ray Image DataSet" [--limit 500]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent / "backend"
sys.path.insert(0, str(BACKEND))

from app.services import fractal_fast as ff  # noqa: E402
from app.services.cnn_model import model as cnn  # noqa: E402
from app.services.preprocessing import preprocess  # noqa: E402
from app.services.segmentation import bounding_box, segmenter  # noqa: E402

CLASSES = {"No_findings": 0, "Pneumonia": 1}


def fractal_vector(roi: np.ndarray) -> np.ndarray:
    """Same 11 numbers, same order as FractalFeatures.to_vector() in fractal.py."""
    g = roi.astype(np.float64)
    binary = g > g.mean()
    bc, _ = ff.box_counting_fd(binary)
    db, _ = ff.dbc_fd(g)
    lac = dict(ff.lacunarity_curve(binary))
    _, _, width = ff.multifractal(g)
    fd_map = ff.local_fd_map(g)
    v = [bc, db, lac[16], width, float(np.nanmean(fd_map)), float(np.nanstd(fd_map)),
         lac[4], lac[8], lac[16], lac[32], lac[64]]
    return np.nan_to_num(np.asarray(v, dtype=np.float32))


def extract(path: str):
    data = open(path, "rb").read()
    pre = preprocess(data, os.path.basename(path))
    gray = pre.gray_full
    roi = gray
    try:
        mask = segmenter.lung_mask(gray)
        if mask is not None and mask.sum() / mask.size > 0.03:
            r0, r1, c0, c1 = bounding_box(mask)
            roi = gray[r0:r1, c0:c1]
    except Exception:
        pass
    fvec = fractal_vector(roi)
    res = cnn.predict(pre.cnn_input)
    return res.embedding.astype(np.float32), fvec, float(res.probabilities.get("Pneumonia", np.nan))


def build_cache(data: Path, limit: int | None, cache: Path):
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in z.files}
    emb, fv, zs, y, grp, names = [], [], [], [], [], []
    t0 = time.time()
    for folder, lab in CLASSES.items():
        files = sorted(f for f in (data / folder).iterdir() if f.suffix.lower() in {".png", ".jpg", ".jpeg"})
        if limit:
            files = files[:limit]
        for i, f in enumerate(files):
            try:
                e, v, z = extract(str(f))
            except Exception as ex:
                print("skip", f.name, ex)
                continue
            emb.append(e); fv.append(v); zs.append(z); y.append(lab)
            grp.append(f.name.split("_")[0]); names.append(f.name)
            if len(y) % 25 == 0:
                print(f"  {len(y)} images  {time.time()-t0:.0f}s", flush=True)
    out = dict(emb=np.stack(emb), fv=np.stack(fv), zs=np.array(zs), y=np.array(y),
               grp=np.array(grp), names=np.array(names))
    np.savez_compressed(cache, **out)
    return out


def fit_mlp(torch, X, y, hidden=32, epochs=60, lr=1e-3, wd=0.1, drop=0.5, seed=0):
    """Small, strongly-regularised MLP. (Our first version trained 300 epochs with weak
    regularisation: it memorised the data -> logits of +-18 -> every new image scored ~100 %.)"""
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.ReLU(),
                              torch.nn.Dropout(drop), torch.nn.Linear(hidden, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    Xt = torch.tensor(X, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.float32)[:, None]
    lossf = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        net.train(); opt.zero_grad(); lossf(net(Xt), yt).backward(); opt.step()
    net.eval()
    return net


def logits(torch, net, X):
    with torch.no_grad():
        return net(torch.tensor(X, dtype=torch.float32))[:, 0].numpy()


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


CONFIGS = [dict(epochs=30), dict(epochs=60), dict(epochs=120)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cache", default=str(HERE / "fusion_features.npz"))
    ap.add_argument("--out", default=str(HERE / "fusion_head_pneumonia_experiment.pt"))
    ap.add_argument("--report", default=str(HERE.parent / "analysis" / "results" / "fusion_results.json"))
    a = ap.parse_args()

    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, roc_auc_score
    from sklearn.model_selection import GroupKFold
    torch.set_num_threads(2)

    if not Path(a.cache).exists():
        assert cnn.available, "TorchXRayVision model not available"
    d = build_cache(Path(a.data), a.limit, Path(a.cache))
    emb, fv, zs, y, grp = d["emb"], d["fv"], d["zs"], d["y"], d["grp"]
    print("features:", emb.shape, fv.shape, "pos rate", y.mean().round(3))

    def stdz(Xtr, Xte):
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        return (Xtr - mu) / sd, (Xte - mu) / sd

    sets = {"B_cnn_head": emb, "C_cnn_plus_fractal": np.hstack([emb, fv]), "fractal_only": fv}
    folds = list(GroupKFold(n_splits=5).split(emb, y, grp))
    oof = {}   # (set, config_idx) -> out-of-fold LOGITS
    for k, X in sets.items():
        for ci, cfg in enumerate(CONFIGS):
            o = np.zeros(len(y))
            for tr, te in folds:
                Xtr, Xte = stdz(X[tr], X[te])
                hid = 16 if k == "fractal_only" else 32
                o[te] = np.mean([logits(torch, fit_mlp(torch, Xtr, y[tr], hidden=hid, seed=s, **cfg), Xte) for s in range(3)], 0)
            oof[(k, ci)] = o
    # pick the config with the best out-of-fold log-loss per feature set (after Platt calibration)
    best, calib = {}, {}
    for k in sets:
        scores = []
        for ci in range(len(CONFIGS)):
            o = oof[(k, ci)]
            lr_ = LogisticRegression(C=1e6).fit(o[:, None], y)
            p = lr_.predict_proba(o[:, None])[:, 1]
            scores.append((log_loss(y, p), ci, float(lr_.coef_[0, 0]), float(lr_.intercept_[0])))
        ll, ci, A, B = min(scores)
        best[k] = ci; calib[k] = (A, B)
        print(f"{k:20s} best epochs={CONFIGS[ci]['epochs']}  OOF logloss {ll:.3f}  platt a={A:.3f} b={B:.3f}")

    def metrics(k):
        o = oof[(k, best[k])]; A, B = calib[k]; p = sigmoid(A * o + B); pr = p >= 0.5
        tp = ((pr == 1) & (y == 1)).sum(); tn = ((pr == 0) & (y == 0)).sum()
        return dict(AUC=float(roc_auc_score(y, o)), accuracy=float((tp + tn) / len(y)),
                    sensitivity=float(tp / (y == 1).sum()), specificity=float(tn / (y == 0).sum()),
                    logloss=float(log_loss(y, p)), epochs=CONFIGS[best[k]]["epochs"])

    rng = np.random.default_rng(0)
    pats = np.unique(grp); idx_by = {p: np.where(grp == p)[0] for p in pats}
    oB, oC = oof[("B_cnn_head", best["B_cnn_head"])], oof[("C_cnn_plus_fractal", best["C_cnn_plus_fractal"])]
    diffs = []
    for _ in range(500):
        ii = np.concatenate([idx_by[p] for p in rng.choice(pats, len(pats))])
        if len(np.unique(y[ii])) == 2:
            diffs.append(roc_auc_score(y[ii], oC[ii]) - roc_auc_score(y[ii], oB[ii]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    results = {
        "n_images": int(len(y)), "n_patients": int(len(pats)),
        "A_cnn_zero_shot": {"AUC": float(roc_auc_score(y, zs))},
        "fractal_only_mlp": metrics("fractal_only"),
        "B_cnn_head": metrics("B_cnn_head"),
        "C_cnn_plus_fractal": metrics("C_cnn_plus_fractal"),
        "AUC_gain_C_minus_B": {"mean": float(np.mean(diffs)), "ci95": [float(lo), float(hi)]},
    }
    print(json.dumps(results, indent=2))
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(results, indent=2))

    # ---- final fusion head (C) on ALL data --------------------------------------------------
    # Fold (1) input standardisation into layer 1 and (2) Platt calibration into the last layer,
    # so the exported net is a plain Linear-ReLU-Dropout-Linear that outputs CALIBRATED logits.
    X = np.hstack([emb, fv]); mu, sd = X.mean(0), X.std(0) + 1e-6
    cfg = CONFIGS[best["C_cnn_plus_fractal"]]
    net = fit_mlp(torch, (X - mu) / sd, y, seed=0, **cfg)
    A, B = calib["C_cnn_plus_fractal"]
    before = sigmoid(A * logits(torch, net, ((X - mu) / sd)[:100]) + B)
    with torch.no_grad():
        W = net[0].weight.clone(); b = net[0].bias.clone()
        net[0].weight.copy_(W / torch.tensor(sd)[None, :])
        net[0].bias.copy_(b - (W * torch.tensor(mu / sd)[None, :]).sum(1))
        net[3].weight.mul_(A); net[3].bias.copy_(net[3].bias * A + B)
    after = sigmoid(logits(torch, net, X[:100]))
    assert np.allclose(before, after, atol=1e-3), "folding changed predictions"
    allp = sigmoid(logits(torch, net, X))
    print("fold check OK | in-sample prob range %.3f..%.3f | mean(normal) %.2f mean(pneu) %.2f"
          % (allp.min(), allp.max(), allp[y == 0].mean(), allp[y == 1].mean()))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "in_dim": X.shape[1], "hidden": 32,
                "labels": ["Pneumonia"]}, a.out)
    print("saved fusion head ->", a.out, "| in_dim", X.shape[1])


if __name__ == "__main__":
    main()
