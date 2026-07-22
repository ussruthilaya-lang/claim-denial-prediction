"""
Report/notebook figures for Phase 4. One consistent visual system across every
chart (validated categorical palette, status colours for risk bands, a single
sequential blue for magnitude), so the deck and the paper read as one product.

Every function saves a PNG under artifacts/figures/ and returns the path, so the
notebook and the report pull the exact same images.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve

from phase4_rag_agentic.src.features import RETRIEVAL_FEATURE_NAMES
from phase4_rag_agentic.src.pipeline import (ARTIFACT_DIR, COST_FN, COST_FP,
                                             Phase4Artifacts)

FIG_DIR = ARTIFACT_DIR / "figures"

# --- validated palette (dataviz skill reference instance) ---
BLUE, GREEN, MAGENTA, YELLOW = "#2a78d6", "#008300", "#e87ba4", "#eda100"
GOOD, WARNING, CRITICAL = "#0ca30c", "#fab219", "#d03b3b"
INK, SECOND, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

mpl.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 140, "figure.facecolor": "white",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": SECOND, "ytick.color": SECOND, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
})


def _save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_ablation(metrics: dict) -> Path:
    """Horizontal AUROC bars per model, with the oracle ceiling as a reference
    line and the leaky bar flagged as invalid."""
    order = [("Structured XGBoost", "B_structured_xgb", BLUE),
             ("+ Retrieval features", "B_retrieval_augmented", GREEN),
             ("Leaky index (invalid)", "B_leaky_index_has_test", MUTED)]
    rows = [(lbl, metrics[key]["auroc"], c) for lbl, key, c in order if key in metrics]
    labels, aurocs, colors = zip(*rows)
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    y = np.arange(len(labels))
    ax.barh(y, aurocs, color=colors, height=0.6, zorder=3)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    for yi, a in zip(y, aurocs):
        ax.text(a - 0.01, yi, f"{a:.3f}", va="center", ha="right",
                color="white", fontweight="bold")
    ceiling = metrics["oracle_ceiling_auroc"]
    ax.axvline(ceiling, color=CRITICAL, lw=2, ls="--", zorder=4)
    ax.text(ceiling + 0.006, 1.0, f"oracle ceiling {ceiling:.3f}", color=CRITICAL,
            va="center", ha="left", rotation=90, fontsize=9, fontweight="bold")
    ax.axvline(0.5, color=MUTED, lw=1, ls=":")
    ax.text(0.5 - 0.006, 0.0, "chance", color=MUTED, fontsize=8,
            va="center", ha="right", rotation=90)
    ax.set_xlim(0.45, 1.02)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.set_xlabel("Test AUROC")
    ax.set_title("Phase 4 ablation: retrieval lift, ceiling, and the leakage trap",
                 pad=12)
    ax.grid(axis="y", visible=False)
    return _save(fig, "ablation_auroc.png")


def plot_calibration(art: Phase4Artifacts) -> Path:
    y = art.test_frame["denied"].to_numpy()
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot([0, 1], [0, 1], color=MUTED, ls=":", lw=1, label="perfect")
    for prob_col, color, name in [("prob_structured", BLUE, "structured"),
                                  ("prob_augmented", GREEN, "+ retrieval")]:
        frac_pos, mean_pred = calibration_curve(
            y, art.test_frame[prob_col].to_numpy(), n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, "-o", color=color, lw=2, ms=6, label=name)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed denial fraction")
    ax.set_title("Calibration: predicted risk vs reality")
    ax.legend(frameon=False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    return _save(fig, "calibration.png")


def plot_cost_curve(art: Phase4Artifacts) -> Path:
    y = art.test_frame["denied"].to_numpy()
    p = art.test_frame["prob_augmented"].to_numpy()
    ts = np.linspace(0.02, 0.98, 97)
    costs = []
    for t in ts:
        pred = (p >= t).astype(int)
        fn = ((pred == 0) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        costs.append((COST_FN * fn + COST_FP * fp) / len(y))
    costs = np.array(costs)
    best_i = int(costs.argmin())
    do_nothing = COST_FN * (y == 1).sum() / len(y)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(ts, costs, color=BLUE, lw=2, zorder=3)
    ax.scatter([ts[best_i]], [costs[best_i]], color=GOOD, s=80, zorder=5,
               label=f"optimal t={ts[best_i]:.2f}  ${costs[best_i]:.2f}/claim")
    ax.axhline(do_nothing, color=CRITICAL, ls="--", lw=1.5,
               label=f"do-nothing ${do_nothing:.2f}/claim")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Expected cost ($/claim)")
    ax.set_title(f"Cost-sensitive operating point (FN=${COST_FN:.0f}, FP=${COST_FP:.0f})")
    ax.legend(frameon=False, fontsize=9)
    return _save(fig, "cost_curve.png")


def plot_shap(art: Phase4Artifacts, sample: int = 1500) -> Path:
    import shap
    from phase4_rag_agentic.src.data_gen import ICD10  # noqa: keep import graph obvious
    test = art.test_frame
    base = art.encoder.transform(test)
    emb = art.encoder.transform_embedding(test)
    retr = art.featurizer.transform(emb, test, exclude_self=False)
    Xa = np.hstack([base, retr])
    idx = np.random.default_rng(0).choice(len(Xa), min(sample, len(Xa)), replace=False)
    expl = shap.TreeExplainer(art.model_augmented)
    sv = expl.shap_values(Xa[idx])
    mean_abs = np.abs(sv).mean(axis=0)
    names = np.array(art.aug_names)
    top = np.argsort(mean_abs)[::-1][:12][::-1]
    colors = [GREEN if names[i] in RETRIEVAL_FEATURE_NAMES else BLUE for i in top]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.barh(np.arange(len(top)), mean_abs[top], color=colors, zorder=3)
    ax.set_yticks(np.arange(len(top)), names[top])
    ax.set_xlabel("mean |SHAP| (impact on denial-risk log-odds)")
    ax.set_title("What the augmented model uses (green = retrieval features)")
    ax.grid(axis="y", visible=False)
    return _save(fig, "shap_bar.png")


def plot_noise_sweep(df) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(df["label_noise"], df["oracle_ceiling"], "-o", color=MUTED,
            lw=1.5, ms=5, label="oracle ceiling")
    ax.plot(df["label_noise"], df["auroc_augmented"], "-o", color=GREEN,
            lw=2, ms=6, label="+ retrieval")
    ax.plot(df["label_noise"], df["auroc_structured"], "-o", color=BLUE,
            lw=2, ms=6, label="structured")
    ax.axhline(0.5, color=MUTED, ls=":", lw=1)
    ax.set_xlabel("Fraction of labels randomly flipped")
    ax.set_ylabel("Test AUROC")
    ax.set_title("Label-noise robustness: how shaky are the injected labels?")
    ax.legend(frameon=False, fontsize=9)
    return _save(fig, "noise_sweep.png")


def plot_harmonization(report_df) -> Path:
    d = report_df[report_df["type"].isin(["numeric", "categorical"])].copy()
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    colors = [CRITICAL if p > 0.25 else (WARNING if p > 0.10 else GOOD)
              for p in d["psi"]]
    y = np.arange(len(d))
    ax.barh(y, d["psi"], color=colors, zorder=3)
    ax.set_yticks(y, d["feature"])
    ax.axvline(0.10, color=MUTED, ls=":", lw=1)
    ax.axvline(0.25, color=CRITICAL, ls="--", lw=1)
    ax.text(0.10, len(d) - 0.4, "0.10", color=MUTED, fontsize=8)
    ax.text(0.25, len(d) - 0.4, "0.25", color=CRITICAL, fontsize=8)
    ax.set_xlabel("Population Stability Index (train vs test)")
    ax.set_title("Feature harmonization: is the population stable?")
    ax.grid(axis="y", visible=False)
    return _save(fig, "harmonization_psi.png")
