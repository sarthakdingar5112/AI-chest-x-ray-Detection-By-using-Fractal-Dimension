# Training — the fusion head

The only model we train. Everything else (the DenseNet CNN) is pretrained via
TorchXRayVision. `train_fusion.py` learns a small MLP that fuses the CNN's 1024-d embedding
with the ~11-dim fractal feature vector to refine multi-label predictions.

## Data
- **NIH ChestX-ray14** (Kaggle: `nih-chest-xrays/data`) — images + `Data_Entry_2017.csv`.
- Any folder of chest X-rays plus a CSV with a `filename` column and one 0/1 column per
  TorchXRayVision pathology (e.g. `Pneumonia`, `Effusion`, `Cardiomegaly`, ...).

## Run (Colab recommended)
```bash
pip install torchxrayvision scikit-image scikit-learn
python train_fusion.py \
  --images  /path/to/images \
  --labels  /path/to/labels.csv \
  --out     ../backend/weights/fusion_head.pt \
  --limit   4000            # subset that fits one Colab session
```

The script prints **per-pathology AUC for CNN-only vs fusion** — that delta is the headline
result for your report/viva. It then exports `fusion_head.pt`, which the backend loads
automatically. If you never run this, the app still works (CNN-only + full fractal analysis).

## Checkpoint format
`{ state_dict, in_dim, hidden, labels }` — consumed by `backend/app/services/fusion.py`.
