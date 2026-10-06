"""
STEP 2 of the dataset study: statistics + figures from the features extracted in step 1.

Usage
-----
    python analysis/analyze.py --data "path/to/X-Ray Image DataSet" --out analysis/results

Answers the examiner's question:  "If a disease occurs, how does the structure (fractal
features) look - and how does it look when there is no disease?"

Produces (in --out):
    fig1_feature_distributions.png   box plots, Normal vs Pneumonia (COVID shown as reference)
    fig2_mean_fd_maps.png            average local-FD map of each class + difference map
    fig3_example_xrays.png           real Normal vs Pneumonia X-rays with their FD heatmaps
    fig4_loglog_box_counting.png     the log-log plots whose slope IS the fractal dimension
    fig5_roc_and_importance.png      how well fractal features alone separate the classes
    table_feature_stats.csv          mean +/- SD, p-value, effect size, AUC per feature
    classifier_results.csv           cross-validated performance of each feature set
    summary.json                     headline numbers
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "savefig.bbox": "tight"})
COL = {"Normal": "#2a9d8f", "Pneumonia": "#e76f51", "COVID-19": "#8d99ae"}

FRACTAL = ["box_counting_fd", "dbc_fd", "lacunarity", "lacunarity_4", "lacunarity_8",
           "lacunarity_32", "lacunarity_64", "multifractal_width", "local_fd_mean", "local_fd_std"]
INTENSITY = ["intensity_mean", "intensity_std"]
NICE = {
    "box_counting_fd": "Box-counting FD", "dbc_fd": "DBC surface FD", "lacunarity": "Lacunarity (r=16)",
    "lacunarity_4": "Lacunarity (r=4)", "lacunarity_8": "Lacunarity (r=8)",
    "lacunarity_32": "Lacunarity (r=32)", "lacunarity_64": "Lacunarity (r=64)",
    "multifractal_width": "Multifractal width", "local_fd_mean": "Local FD (mean)",
    "local_fd_std": "Local FD (std)", "intensity_mean": "Mean brightness", "intensity_std": "Brightness SD",
}


def holm(p: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values."""
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(ranked, 1.0)
    return out


def cohens_d(a, b):
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (b.mean() - a.mean()) / sp


# ------------------------------------------------------------------ statistics table
def feature_table(df: pd.DataFrame) -> pd.DataFrame:
    n = df[df.label == "Normal"]
    p = df[df.label == "Pneumonia"]
    c = df[df.label == "COVID-19"]
    rows = []
    for f in FRACTAL + INTENSITY:
        u = mannwhitneyu(n[f], p[f])
        auc = roc_auc_score(np.r_[np.zeros(len(n)), np.ones(len(p))], np.r_[n[f], p[f]])
        rows.append({
            "feature": NICE[f], "key": f,
            "Normal (mean ± SD)": f"{n[f].mean():.3f} ± {n[f].std():.3f}",
            "Pneumonia (mean ± SD)": f"{p[f].mean():.3f} ± {p[f].std():.3f}",
            "COVID-19 (mean ± SD)": f"{c[f].mean():.3f} ± {c[f].std():.3f}",
            "change in pneumonia": "higher" if p[f].mean() > n[f].mean() else "lower",
            "% change": 100 * (p[f].mean() - n[f].mean()) / abs(n[f].mean()),
            "Cohen d": cohens_d(n[f], p[f]),
            "AUC (single feature)": max(auc, 1 - auc),
            "p_raw": u.pvalue,
        })
    t = pd.DataFrame(rows)
    t["p (Holm-adjusted)"] = holm(t.p_raw.values)
    t["significant"] = np.where(t["p (Holm-adjusted)"] < 0.05, "yes", "no")
    return t


# ------------------------------------------------------------------ figures
def fig_distributions(df, path):
    keys = ["dbc_fd", "local_fd_mean", "lacunarity", "lacunarity_4", "local_fd_std", "box_counting_fd"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2))
    for ax, k in zip(axes.ravel(), keys):
        data = [df[df.label == l][k].values for l in ("Normal", "Pneumonia", "COVID-19")]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False,
                        medianprops=dict(color="black"))
        for patch, l in zip(bp["boxes"], ("Normal", "Pneumonia", "COVID-19")):
            patch.set_facecolor(COL[l]); patch.set_alpha(0.85 if l != "COVID-19" else 0.35)
        ax.set_xticks([1, 2, 3], ["Normal", "Pneumonia", "COVID-19*"])
        ax.set_title(NICE[k], fontsize=10.5)
    fig.suptitle("How the lung texture differs: Normal vs Pneumonia chest X-rays "
                 f"(n = {int((df.label=='Normal').sum())} + {int((df.label=='Pneumonia').sum())})",
                 fontweight="bold")
    fig.text(0.5, -0.01, "*COVID-19 images come from a different source/scanner mix - shown for reference, "
             "not used in the statistical comparison.", ha="center", fontsize=8.5, color="#555")
    fig.tight_layout()
    fig.savefig(path); plt.close(fig)


def fig_mean_maps(npz, path):
    maps, labels = npz["maps"], npz["labels"]
    mn = maps[labels == "Normal"]; mp = maps[labels == "Pneumonia"]
    mean_n, mean_p = mn.mean(0), mp.mean(0)
    diff = mean_p - mean_n
    # per-cell significance (Mann-Whitney) with FDR
    pv = np.array([[mannwhitneyu(mn[:, i, j], mp[:, i, j]).pvalue for j in range(mn.shape[2])]
                   for i in range(mn.shape[1])])
    sig = fdr(pv.ravel()).reshape(pv.shape) < 0.05
    vmin, vmax = min(mean_n.min(), mean_p.min()), max(mean_n.max(), mean_p.max())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), gridspec_kw={"wspace": 0.45})
    for ax, m, t in ((axes[0], mean_n, f"NORMAL - average local FD map (n={len(mn)})"),
                     (axes[1], mean_p, f"PNEUMONIA - average local FD map (n={len(mp)})")):
        im = ax.imshow(m, cmap="turbo", vmin=vmin, vmax=vmax); ax.set_title(t, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes[:2], fraction=0.025, pad=0.03, label="local fractal dimension")
    lim = np.abs(diff).max()
    im2 = axes[2].imshow(diff, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ys, xs = np.where(sig)
    axes[2].scatter(xs, ys, s=14, c="k", marker=".")
    axes[2].set_title("DIFFERENCE (Pneumonia - Normal)\n• = statistically significant (FDR < 0.05)", fontsize=10)
    axes[2].set_xticks([]); axes[2].set_yticks([])
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.03, label="Δ local FD")
    fig.savefig(path); plt.close(fig)
    return float(sig.mean()), diff


def pick_examples(df, label, k=3):
    sub = df[df.label == label].copy()
    med = sub.dbc_fd.median()
    # typical examples: closest to the class median in DBC FD, spread out a little
    sub["dist"] = (sub.dbc_fd - med).abs()
    return sub.sort_values("dist").head(k * 6).iloc[::6].head(k)


def fig_examples(df, npz, data_dir: Path, path, k=3):
    folder = {"Normal": "No_findings", "Pneumonia": "Pneumonia"}
    maps = {f: m for f, m in zip(npz["files"], npz["maps"])}
    fig, axes = plt.subplots(2, 2 * k, figsize=(3 * 2 * k, 7.2), gridspec_kw={"hspace": 0.25, "wspace": 0.05})
    for r, label in enumerate(("Normal", "Pneumonia")):
        ex = pick_examples(df, label, k)
        for c, (_, row) in enumerate(ex.iterrows()):
            g = np.asarray(cv2.imread(str(data_dir / folder[label] / row.file), cv2.IMREAD_GRAYSCALE))
            g = cv2.createCLAHE(2.0, (8, 8)).apply(cv2.resize(g, (512, 512), interpolation=cv2.INTER_AREA))
            m = maps[row.file]
            mm = cv2.resize(m.astype(np.float32), (512, 512), interpolation=cv2.INTER_CUBIC)
            # the map covers window centres; stretch to the image for display
            ax = axes[r, 2 * c]; ax.imshow(g, cmap="gray"); ax.axis("off")
            ax.set_title(f"{label}\nDBC FD = {row.dbc_fd:.3f}", fontsize=9.5, color=COL[label], fontweight="bold")
            ax = axes[r, 2 * c + 1]; ax.imshow(g, cmap="gray")
            ax.imshow(mm, cmap="turbo", alpha=0.55, vmin=1.9, vmax=2.4); ax.axis("off")
            ax.set_title(f"local FD map\nmean = {row.local_fd_mean:.3f}", fontsize=9.5)
    fig.suptitle("Real examples: what 'no disease' vs 'disease' looks like through fractal analysis "
                 "(blue = smooth texture, red = rough / complex texture)", fontweight="bold", fontsize=11, y=0.98)
    fig.savefig(path); plt.close(fig)


def fig_loglog(cz, path):
    labels = cz["labels"]
    sizes = cz["dbc_sizes"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, key, szkey, xl, title in ((axes[0], "dbc", "dbc_sizes", "log(1/r)", "DBC (grey-level surface)"),
                                      (axes[1], "box", "box_sizes", "log(1/r)", "Box counting (binary)")):
        sz = cz[szkey]
        for l in ("Normal", "Pneumonia"):
            y = np.log(cz[key][labels == l]).mean(0)
            x = np.log(1.0 / sz)
            slope, icpt = np.polyfit(x, y, 1)
            ax.plot(x, y, "o", color=COL[l])
            ax.plot(x, slope * x + icpt, "-", color=COL[l], label=f"{l}: slope = FD = {slope:.3f}")
        ax.set_xlabel(xl); ax.set_ylabel("log N(r)  (number of boxes)")
        ax.set_title(title); ax.legend(frameon=False)
    fig.suptitle("The fractal dimension is the slope of the log-log line (class averages)", fontweight="bold")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# ------------------------------------------------------------------ classification
def groups_from_files(files: pd.Series) -> np.ndarray:
    """NIH file names look like 00000061_012.png -> patient 00000061. Keeps one patient in one fold."""
    return files.str.split("_").str[0].values


def classify(df, path_fig):
    d = df[df.label.isin(["Normal", "Pneumonia"])].reset_index(drop=True)
    y = (d.label == "Pneumonia").astype(int).values
    groups = groups_from_files(d.file)
    cv = GroupKFold(n_splits=5)
    sets = {
        "Brightness only (control)": INTENSITY,
        "Fractal features only": FRACTAL,
        "Fractal + brightness": FRACTAL + INTENSITY,
    }
    models = {
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "Random forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=0, n_jobs=2),
    }
    rows, curves = [], {}
    for sname, cols in sets.items():
        for mname, mdl in models.items():
            prob = cross_val_predict(mdl, d[cols], y, cv=cv, groups=groups, method="predict_proba")[:, 1]
            pred = (prob >= 0.5).astype(int)
            tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
            rows.append({"features": sname, "model": mname, "AUC": roc_auc_score(y, prob),
                         "accuracy": (tp + tn) / len(y), "sensitivity": tp / (tp + fn),
                         "specificity": tn / (tn + fp), "n_features": len(cols)})
            curves[(sname, mname)] = (prob, roc_curve(y, prob))
    res = pd.DataFrame(rows)

    # importance (RF on fractal only, fit on everything - for ranking only)
    rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=3, random_state=0, n_jobs=2).fit(d[FRACTAL], y)
    imp = pd.Series(rf.feature_importances_, index=[NICE[c] for c in FRACTAL]).sort_values()

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax = axes[0]
    styles = {"Brightness only (control)": ("#999", "--"), "Fractal features only": ("#e76f51", "-"),
              "Fractal + brightness": ("#264653", "-")}
    for sname, (c, ls) in styles.items():
        prob, (fpr, tpr, _) = curves[(sname, "Random forest")]
        a = res[(res.features == sname) & (res.model == "Random forest")].AUC.iloc[0]
        ax.plot(fpr, tpr, color=c, ls=ls, lw=2, label=f"{sname}  (AUC {a:.2f})")
    ax.plot([0, 1], [0, 1], ":", color="#bbb")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("Normal vs Pneumonia - 5-fold patient-grouped CV (Random forest)", fontsize=10)
    ax.legend(frameon=False, loc="lower right")
    axes[1].barh(imp.index, imp.values, color="#e76f51")
    axes[1].set_title("Which fractal features matter most?", fontsize=10)
    axes[1].set_xlabel("Random-forest importance")
    fig.tight_layout(); fig.savefig(path_fig); plt.close(fig)
    return res, imp


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=str(HERE / "results"))
    a = ap.parse_args()
    out = Path(a.out); data_dir = Path(a.data)

    df = pd.read_csv(out / "features.csv")
    npz = np.load(out / "local_fd_maps.npz", allow_pickle=True)
    cz = np.load(out / "curves.npz", allow_pickle=True)

    table = feature_table(df)
    table.drop(columns=["p_raw"]).to_csv(out / "table_feature_stats.csv", index=False)
    fig_distributions(df, out / "fig1_feature_distributions.png")
    frac_sig, diff = fig_mean_maps(npz, out / "fig2_mean_fd_maps.png")
    fig_examples(df, npz, data_dir, out / "fig3_example_xrays.png")
    fig_loglog(cz, out / "fig4_loglog_box_counting.png")
    res, imp = classify(df, out / "fig5_roc_and_importance.png")
    res.round(4).to_csv(out / "classifier_results.csv", index=False)

    ft = table[table.key.isin(FRACTAL)]
    best = ft.sort_values("AUC (single feature)", ascending=False).iloc[0]
    rf_frac = res[(res.features == "Fractal features only") & (res.model == "Random forest")].iloc[0]
    rf_int = res[(res.features == "Brightness only (control)") & (res.model == "Random forest")].iloc[0]
    rf_both = res[(res.features == "Fractal + brightness") & (res.model == "Random forest")].iloc[0]
    summary = {
        "n_normal": int((df.label == "Normal").sum()), "n_pneumonia": int((df.label == "Pneumonia").sum()),
        "n_covid_reference": int((df.label == "COVID-19").sum()),
        "significant_fractal_features": int((ft["significant"] == "yes").sum()), "total_fractal_features": len(ft),
        "best_single_feature": best.feature, "best_single_feature_auc": round(float(best["AUC (single feature)"]), 3),
        "best_single_feature_change_pct": round(float(best["% change"]), 2),
        "rf_fractal_only": {k: round(float(rf_frac[k]), 3) for k in ("AUC", "accuracy", "sensitivity", "specificity")},
        "rf_brightness_only": {k: round(float(rf_int[k]), 3) for k in ("AUC", "accuracy", "sensitivity", "specificity")},
        "rf_fractal_plus_brightness": {k: round(float(rf_both[k]), 3) for k in ("AUC", "accuracy", "sensitivity", "specificity")},
        "fraction_of_map_cells_significant": round(frac_sig, 3),
        "top_importance": imp.sort_values(ascending=False).head(3).index.tolist(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(table[["feature", "Normal (mean ± SD)", "Pneumonia (mean ± SD)", "% change", "Cohen d",
                 "AUC (single feature)", "p (Holm-adjusted)", "significant"]].round(4).to_string(index=False))
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
