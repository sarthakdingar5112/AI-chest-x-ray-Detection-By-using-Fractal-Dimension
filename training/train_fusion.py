"""
Train the fusion head:  [CNN embedding (1024) || fractal vector (~11)] -> pathology logits.

This is the one training step of the project. It stands on TorchXRayVision's pretrained
DenseNet (no CNN training from scratch) and only learns a small MLP that lets the fractal
features refine the CNN's multi-label predictions.

USAGE (Colab or local GPU)
--------------------------
1. Get the NIH ChestX-ray14 dataset (Kaggle: `nih-chest-xrays/data`), or any folder of
   chest X-rays with a CSV of multi-label targets aligned to TorchXRayVision pathologies.
2. Point --images at the image folder and --labels at the CSV.
3. Run:  python train_fusion.py --images /path/imgs --labels /path/labels.csv --out ../backend/weights/fusion_head.pt

The exported checkpoint contains: {state_dict, in_dim, hidden, labels}. `backend/app/services/
fusion.py` loads exactly this format. If you skip training, the app runs CNN-only + fractal.

COLAB QUICKSTART
----------------
    !pip install torchxrayvision scikit-image
    # upload this file + backend/app/services/fractal.py, then run the CLI above.
    # Finally download backend/weights/fusion_head.pt and drop it into the repo.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# Make the backend fractal core importable.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
from app.services.fractal import compute_fractal_features  # noqa: E402


def extract_features(model, xrv, torch, img_path: str):
    """Return (cnn_embedding[1024], fractal_vector[~11], cnn_probs[dict])."""
    from skimage.io import imread
    img = imread(img_path)
    if img.ndim == 3:
        img = img.mean(axis=2)
    img = img.astype("float32")

    # CNN branch (TorchXRayVision expects [-1024,1024], 224x224, single channel).
    xr = xrv.datasets.normalize(img, 255)
    xr = xrv.datasets.XRayCenterCrop()(xr[None, ...])
    xr = xrv.datasets.XRayResizer(224)(xr)
    t = torch.from_numpy(xr)[None, ...]
    acts = {}
    h = model.features.register_forward_hook(lambda m, i, o: acts.__setitem__("f", o.detach()))
    with torch.no_grad():
        logits = model(t)
    h.remove()
    emb = acts["f"].mean(dim=[2, 3])[0].numpy()

    # Fractal branch.
    fvec = compute_fractal_features(img).to_vector()
    return emb, fvec, logits[0].numpy()


def label_vector(row: dict, labels: list) -> list:
    """
    Build a 0/1 target vector for the TorchXRayVision `labels`.

    Supports two CSV formats:
      - NIH native (`Data_Entry_2017.csv`): a single 'Finding Labels' column with
        pipe-separated names, e.g. "Cardiomegaly|Effusion".
      - One-hot: one 0/1 column per pathology name.
    """
    if "Finding Labels" in row:
        present = {s.strip() for s in (row["Finding Labels"] or "").split("|")}
        return [1.0 if l in present else 0.0 for l in labels]
    return [float(row.get(l, 0) or 0) for l in labels]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder of chest X-ray images")
    ap.add_argument("--labels", required=True, help="CSV: filename + one column per pathology (0/1)")
    ap.add_argument("--out", default="../backend/weights/fusion_head.pt")
    ap.add_argument("--limit", type=int, default=4000, help="max images (subset for Colab)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=256)
    args = ap.parse_args()

    import csv
    import torch
    import torch.nn as nn
    import torchxrayvision as xrv
    from sklearn.metrics import roc_auc_score

    model = xrv.models.DenseNet(weights="densenet121-res224-all").eval()
    labels = [p for p in model.pathologies if p]

    # Index every image under --images by basename (handles the flat 'sample' layout AND
    # the full NIH dataset, which nests images across images_001/ ... images_012/).
    file_index = {}
    for root, _dirs, files in os.walk(args.images):
        for f in files:
            file_index.setdefault(f, os.path.join(root, f))
    print(f"indexed {len(file_index)} image files under {args.images}")

    # --- build feature matrix --------------------------------------------------------
    rows = list(csv.DictReader(open(args.labels)))[: args.limit]
    X, Y, cnn_only = [], [], []
    for i, row in enumerate(rows):
        fn = row.get("filename") or row.get("Image Index")
        path = file_index.get(fn) or os.path.join(args.images, fn or "")
        if not fn or not os.path.exists(path):
            continue
        try:
            emb, fvec, probs = extract_features(model, xrv, torch, path)
        except Exception as e:
            print("skip", fn, e)
            continue
        X.append(np.concatenate([emb, fvec]))
        Y.append(label_vector(row, labels))
        cnn_only.append(probs[: len(labels)])
        if i % 100 == 0:
            print(f"  processed {i}/{len(rows)}")

    X = np.asarray(X, dtype="float32")
    Y = np.asarray(Y, dtype="float32")
    cnn_only = np.asarray(cnn_only, dtype="float32")
    print("feature matrix:", X.shape, "targets:", Y.shape)

    # --- train MLP -------------------------------------------------------------------
    n = len(X)
    idx = np.random.permutation(n)
    tr, va = idx[: int(0.85 * n)], idx[int(0.85 * n):]
    net = nn.Sequential(nn.Linear(X.shape[1], args.hidden), nn.ReLU(), nn.Dropout(0.3), nn.Linear(args.hidden, len(labels)))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    Xt, Yt = torch.tensor(X[tr]), torch.tensor(Y[tr])
    for ep in range(args.epochs):
        net.train()
        opt.zero_grad()
        loss = lossf(net(Xt), Yt)
        loss.backward()
        opt.step()
        if ep % 5 == 0:
            print(f"epoch {ep:3d}  loss {loss.item():.4f}")

    # --- evaluate: fusion vs CNN-only ------------------------------------------------
    net.eval()
    with torch.no_grad():
        fused = torch.sigmoid(net(torch.tensor(X[va]))).numpy()
    print("\nper-pathology AUC (CNN-only -> fusion):")
    for k, l in enumerate(labels):
        yv = Y[va, k]
        if len(np.unique(yv)) < 2:
            continue
        try:
            a_cnn = roc_auc_score(yv, cnn_only[va, k])
            a_fus = roc_auc_score(yv, fused[:, k])
            print(f"  {l:20s} {a_cnn:.3f} -> {a_fus:.3f}  ({'+' if a_fus>=a_cnn else ''}{a_fus-a_cnn:+.3f})")
        except Exception:
            pass

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "in_dim": X.shape[1], "hidden": args.hidden, "labels": labels}, args.out)
    print("\nsaved fusion head ->", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
