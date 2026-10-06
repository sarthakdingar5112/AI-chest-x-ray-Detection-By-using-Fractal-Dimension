"""
MULTI-DISEASE STUDY: how does lung texture (fractal features) look for each disease vs normal?

Each disease is compared ONLY with 'normal' images from the SAME public source, because
different hospitals/scanners/JPEG pipelines change texture numbers on their own.

    Source A  muhammedtalo/COVID-19            Normal(500)  Pneumonia(500)            [NIH images]
    Source B  education454/datasets            Normal(sample) COVID-19(712)
    Source C  ieee8023/covid-chestxray-dataset Normal + Tuberculosis, Bacterial, Viral, Fungal, COVID-19
                                               (small groups -> exploratory only)

Usage
-----
    python analysis/multi_disease.py --a <A dir> --b <B dir> --c <C dir> [--out analysis/results/multi]
Writes features_multi.csv + figures fig6..fig8 + table_multi_disease.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from extract_features import process  # noqa: E402

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "savefig.bbox": "tight"})

FEATS = ["dbc_fd", "local_fd_mean", "local_fd_std", "lacunarity_4", "lacunarity", "box_counting_fd",
         "multifractal_width"]
NICE = {"dbc_fd": "DBC surface FD", "local_fd_mean": "Local FD (mean)", "local_fd_std": "Local FD (std)",
        "lacunarity_4": "Lacunarity r=4", "lacunarity": "Lacunarity r=16",
        "box_counting_fd": "Box-counting FD", "multifractal_width": "Multifractal width"}


def build_manifest(a: Path, b: Path, c: Path, n_norm_b: int = 800) -> pd.DataFrame:
    rows = []
    # Source A
    for folder, cls in (("No_findings", "Normal"), ("Pneumonia", "Pneumonia")):
        for f in sorted((a / "X-Ray Image DataSet" / folder).iterdir()):
            rows.append((str(f), "A", cls))
    # Source B
    rng = np.random.default_rng(0)
    normals = sorted(p for s in ("train", "test") for p in (b / "Data" / s / "NORMAL").iterdir())
    for f in rng.choice(normals, min(n_norm_b, len(normals)), replace=False):
        rows.append((str(f), "B", "Normal"))
    for s in ("train", "test"):
        for f in sorted((b / "Data" / s / "COVID19").iterdir()):
            rows.append((str(f), "B", "COVID-19"))
    # Source C
    m = pd.read_csv(c / "metadata.csv")
    m = m[(m.modality == "X-ray") & m.view.isin(["PA", "AP", "AP Erect"])]

    def grp(f):
        f = str(f)
        if f == "No Finding": return "Normal"
        if f == "Tuberculosis": return "Tuberculosis"
        if "COVID-19" in f: return "COVID-19"
        if f.startswith("Pneumonia/Bacterial"): return "Bacterial pneumonia"
        if f.startswith("Pneumonia/Fungal"): return "Fungal pneumonia"
        if f.startswith("Pneumonia/Viral"): return "Viral pneumonia (non-COVID)"
        return None
    m["g"] = m.finding.map(grp)
    for _, r in m.dropna(subset=["g"]).iterrows():
        p = c / "images" / r.filename
        if p.exists():
            rows.append((str(p), "C", r.g))
    return pd.DataFrame(rows, columns=["path", "source", "cls"])


def run_one(args):
    path, label = args
    return process((path, label))


def extract(man: pd.DataFrame, out: Path, workers: int):
    items = [(r.path, f"{r.source}|{r.cls}") for r in man.itertuples()]
    rows, maps = [], []
    with Pool(workers) as pool:
        for k, res in enumerate(pool.imap(run_one, items, chunksize=4), 1):
            if res is None:
                continue
            r, m, *_ = res
            r["path"] = items[k - 1][0]
            rows.append(r); maps.append(m)
            if k % 200 == 0:
                print(f"  {k}/{len(items)}", flush=True)
    df = pd.DataFrame(rows)
    df[["source", "cls"]] = df.label.str.split("|", n=1, expand=True)
    df = df.drop(columns="label")
    df.to_csv(out / "features_multi.csv", index=False)
    np.savez_compressed(out / "maps_multi.npz", maps=np.stack(maps), source=df.source.values, cls=df.cls.values,
                        path=df.path.values)
    return df, np.stack(maps)


def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def cohens_d(n, d):
    sp = np.sqrt(((len(n) - 1) * n.var(ddof=1) + (len(d) - 1) * d.var(ddof=1)) / (len(n) + len(d) - 2))
    return (d.mean() - n.mean()) / sp


def compare(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for src, sub in df.groupby("source"):
        norm = sub[sub.cls == "Normal"]
        for cls in sorted(set(sub.cls) - {"Normal"}):
            dis = sub[sub.cls == cls]
            if len(dis) < 8 or len(norm) < 8:
                continue
            for f in FEATS:
                u = mannwhitneyu(norm[f], dis[f])
                rows.append(dict(source=src, disease=cls, n_normal=len(norm), n_disease=len(dis), feature=f,
                                 normal_mean=norm[f].mean(), disease_mean=dis[f].mean(),
                                 pct_change=100 * (dis[f].mean() - norm[f].mean()) / abs(norm[f].mean()),
                                 cohen_d=cohens_d(norm[f], dis[f]), auc=max(u.statistic / (len(norm) * len(dis)),
                                 1 - u.statistic / (len(norm) * len(dis))), p=u.pvalue))
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t.p.values)
    t["significant"] = t.p_holm < 0.05
    return t


SRC = {"A": "NIH-derived", "B": "Kaggle/COVID set", "C": "Cohen collection"}


def fig_effect_heatmap(t, path):
    t = t.copy()
    t["row"] = t.disease + "  [" + t.source.map(SRC) + "]  n=" + t.n_disease.astype(str) + " vs " + t.n_normal.astype(str)
    order = t.drop_duplicates("row").sort_values(["source", "disease"]).row.tolist()
    d = t.pivot(index="row", columns="feature", values="cohen_d").loc[order, FEATS]
    s = t.pivot(index="row", columns="feature", values="significant").loc[order, FEATS]
    fig, ax = plt.subplots(figsize=(10.5, 0.62 * len(order) + 2.3))
    lim = 1.5
    im = ax.imshow(d.values, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            ax.text(j, i, f"{d.values[i, j]:+.2f}" + ("*" if s.values[i, j] else ""), ha="center", va="center",
                    fontsize=9, color="white" if abs(d.values[i, j]) > 0.9 else "black")
    ax.set_xticks(range(len(FEATS)), [NICE[f] for f in FEATS], rotation=30, ha="right")
    ax.set_yticks(range(len(order)), order)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="effect size (Cohen d): disease - normal")
    ax.set_title("How each disease changes lung texture vs its own-source normal group\n"
                 "red = higher in disease, blue = lower   |   * = significant (Holm-corrected p < 0.05)",
                 fontsize=10.5, fontweight="bold")
    fig.savefig(path); plt.close(fig)


def fig_boxes(df, path, feat="dbc_fd"):
    srcs = ["A", "B", "C"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [2, 2, 5]})
    for ax, s in zip(axes, srcs):
        sub = df[df.source == s]
        classes = ["Normal"] + sorted(set(sub.cls) - {"Normal"})
        data = [sub[sub.cls == c][feat].values for c in classes]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False, medianprops=dict(color="black"), widths=0.6)
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor("#2a9d8f" if i == 0 else "#e76f51"); patch.set_alpha(0.85)
        ax.set_xticks(range(1, len(classes) + 1), [f"{c}\n(n={len(d)})".replace(" (non-COVID)", "\n(non-COVID)").replace(" pneumonia", "\npneumonia")
                                                for c, d in zip(classes, data)], fontsize=8)
        ax.set_title(f"Source {s}: {SRC[s]}", fontsize=10)
    axes[0].set_ylabel(NICE[feat])
    fig.suptitle(f"{NICE[feat]} for each disease vs normal (compare boxes only WITHIN a panel)", fontweight="bold")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_diff_maps(df, maps, path):
    panels = []
    for s in ["A", "B", "C"]:
        sm = (df.source == s).values
        nm = sm & (df.cls == "Normal").values
        for c in sorted(set(df[df.source == s].cls) - {"Normal"}):
            cm = sm & (df.cls == c).values
            if cm.sum() >= 8 and nm.sum() >= 8:
                panels.append((s, c, int(cm.sum()), maps[cm].mean(0) - maps[nm].mean(0)))
    n = len(panels); cols = min(4, n); rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.5 * rows))
    axes = np.atleast_1d(axes).ravel()
    lim = max(np.abs(p[3]).max() for p in panels)
    for ax, (s, c, k, dm) in zip(axes, panels):
        im = ax.imshow(dm, cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_title(f"{c}\n[{SRC[s]}] n={k}", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis("off")
    fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.02, label="Δ local FD (disease - own normal)")
    fig.suptitle("WHERE the texture changes: average local-FD difference map per disease "
                 "(red = rougher than normal)", fontweight="bold")
    fig.savefig(path); plt.close(fig)


def fig_examples(df, maps, path):
    items = []
    for s, c in [("A", "Normal"), ("A", "Pneumonia"), ("B", "COVID-19"), ("C", "Tuberculosis"),
                 ("C", "Bacterial pneumonia"), ("C", "Viral pneumonia (non-COVID)"), ("C", "Fungal pneumonia"),
                 ("C", "Normal")]:
        idx = np.where(((df.source == s) & (df.cls == c)).values)[0]
        if len(idx) == 0:
            continue
        med = df.local_fd_mean.values[idx].mean()
        i = idx[np.argmin(np.abs(df.local_fd_mean.values[idx] - med))]   # a typical example
        items.append((s, c, i))
    cols = 4; rows = int(np.ceil(len(items) / cols)) * 2
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.3 * rows), gridspec_kw={"hspace": 0.3, "wspace": 0.04})
    for k, (s, c, i) in enumerate(items):
        r, cc = (k // cols) * 2, k % cols
        g = cv2.imread(df.path.values[i], cv2.IMREAD_GRAYSCALE)
        g = cv2.createCLAHE(2.0, (8, 8)).apply(cv2.resize(g, (512, 512), interpolation=cv2.INTER_AREA))
        mm = cv2.resize(maps[i].astype(np.float32), (512, 512), interpolation=cv2.INTER_CUBIC)
        axes[r, cc].imshow(g, cmap="gray"); axes[r, cc].axis("off")
        axes[r, cc].set_title(f"{c.replace(' (non-COVID)', chr(10) + '(non-COVID)')}\n[{SRC[s]}]", fontsize=8.5, fontweight="bold",
                              color="#2a9d8f" if c == "Normal" else "#e76f51")
        axes[r + 1, cc].imshow(g, cmap="gray"); axes[r + 1, cc].imshow(mm, cmap="turbo", alpha=0.55, vmin=1.9, vmax=2.4)
        axes[r + 1, cc].axis("off"); axes[r + 1, cc].set_title(f"local FD mean {df.local_fd_mean.values[i]:.3f}", fontsize=8.5)
    for ax in axes.ravel():
        if not ax.has_data():
            ax.axis("off")
    fig.subplots_adjust(top=0.92)
    fig.suptitle("A typical example of each condition and its fractal texture map (warm = rough/complex)",
                 fontweight="bold")
    fig.savefig(path); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True); ap.add_argument("--c", required=True)
    ap.add_argument("--out", default=str(HERE / "results" / "multi"))
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--reuse", action="store_true", help="reuse features_multi.csv if present")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.reuse and (out / "features_multi.csv").exists():
        df = pd.read_csv(out / "features_multi.csv"); maps = np.load(out / "maps_multi.npz", allow_pickle=True)["maps"]
    else:
        man = build_manifest(Path(a.a), Path(a.b), Path(a.c))
        print(man.groupby(["source", "cls"]).size().to_string(), flush=True)
        df, maps = extract(man, out, a.workers)
    t = compare(df)
    t.round(5).to_csv(out / "table_multi_disease.csv", index=False)
    fig_effect_heatmap(t, out / "fig6_effect_sizes_per_disease.png")
    fig_boxes(df, out / "fig7_dbc_fd_per_disease.png")
    fig_diff_maps(df, maps, out / "fig8_difference_maps_per_disease.png")
    fig_examples(df, maps, out / "fig9_examples_per_disease.png")
    pd.set_option("display.width", 220)
    print(t[t.feature.isin(["dbc_fd", "local_fd_mean"])][["source", "disease", "n_normal", "n_disease", "feature",
          "normal_mean", "disease_mean", "pct_change", "cohen_d", "auc", "p_holm", "significant"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
