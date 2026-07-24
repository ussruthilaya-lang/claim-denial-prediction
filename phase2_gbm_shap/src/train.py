"""
Phase 2 — gradient boosting + SHAP, on the unified dataset. Replaces Phase 1's
LR/DT with GradientBoosting + RandomForest and adds per-feature SHAP attribution,
so we can say WHICH structured signals drive denial before the later phases add
text (Phase 3) and retrieval (Phase 4). Same StructuredEncoder + temporal split
as every phase, so the AUROC is comparable and the SHAP ranking is over the same
feature space. Notably, SHAP here can only attribute to observable billing
fields — it structurally cannot see the note-only necessity signal, which is
exactly the gap Phase 3 fills.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sklearn.ensemble import (GradientBoostingClassifier,  # noqa: E402
                              RandomForestClassifier)

from phase4_rag_agentic.src.data_gen import generate_dataset  # noqa: E402
from phase4_rag_agentic.src.features import StructuredEncoder  # noqa: E402
from phase4_rag_agentic.src.pipeline import _temporal_split  # noqa: E402
from shared.utils.eval import evaluate  # noqa: E402

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
FIG_DIR = ARTIFACT_DIR / "figures"
PHASE = "phase2"


def _mean_abs_shap(explainer, X: np.ndarray) -> np.ndarray:
    """Reduce SHAP output to mean|SHAP| per feature, robust to the several shapes
    shap returns across versions (list per class / 3-D stack / 2-D)."""
    sv = explainer.shap_values(X)
    sv = np.asarray(sv)
    if sv.ndim == 3:  # (n, feat, class) or (class, n, feat)
        sv = sv[:, :, 1] if sv.shape[-1] == 2 else sv[1]
    return np.abs(sv).mean(axis=0)


def run_phase2(n: int = 40_000, seed: int = 42, shap_sample: int = 2000) -> dict:
    df = generate_dataset(n=n, seed=seed)
    train, test = _temporal_split(df)
    enc = StructuredEncoder().fit(train)
    Xtr, Xte = enc.transform(train), enc.transform(test)
    ytr, yte = train["denied"].to_numpy(), test["denied"].to_numpy()
    names = enc.feature_names

    gbm = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                     learning_rate=0.1, random_state=42).fit(Xtr, ytr)
    rf = RandomForestClassifier(n_estimators=300, max_depth=8, n_jobs=-1,
                                random_state=42).fit(Xtr, ytr)
    r_gbm = evaluate(yte, gbm.predict_proba(Xte)[:, 1], PHASE, "gradient-boosting").as_dict()
    r_rf = evaluate(yte, rf.predict_proba(Xte)[:, 1], PHASE, "random-forest").as_dict()

    # ---- SHAP attribution on the GBM ----
    import shap
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xte), size=min(shap_sample, len(Xte)), replace=False)
    explainer = shap.TreeExplainer(gbm)
    mean_abs = _mean_abs_shap(explainer, Xte[idx])
    order = np.argsort(mean_abs)[::-1]
    top = [(names[i], float(mean_abs[i])) for i in order[:12]]

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    labels = [t[0] for t in top][::-1]
    vals = [t[1] for t in top][::-1]
    plt.figure(figsize=(7, 5))
    plt.barh(labels, vals, color="#4C78A8")
    plt.xlabel("mean |SHAP|  (impact on denial log-odds)")
    plt.title("Phase 2 — structured feature attribution (GBM)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_summary.png", dpi=150)
    plt.close()

    return {
        "phase": PHASE,
        "n_train": int(len(train)), "n_test": int(len(test)),
        "test_prevalence": float(yte.mean()),
        "gradient_boosting": r_gbm,
        "random_forest": r_rf,
        "shap_top_features": top,
    }


def main(n: int = 40_000, seed: int = 42, save: bool = True) -> dict:
    m = run_phase2(n=n, seed=seed)
    print("\n================ PHASE 2 SUMMARY ================")
    print(f"n_train={m['n_train']} n_test={m['n_test']} prevalence={m['test_prevalence']:.3f}")
    print(f"{'model':22s} {'AUROC':>7s} {'F1':>7s} {'P':>7s} {'R':>7s}")
    for key in ["gradient_boosting", "random_forest"]:
        r = m[key]
        print(f"{key:22s} {r['auroc']:7.4f} {r['f1']:7.4f} {r['precision']:7.4f} {r['recall']:7.4f}")
    print("top SHAP features (structured):")
    for name, val in m["shap_top_features"][:6]:
        print(f"  {name:28s} {val:.4f}")
    if save:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        with open(ARTIFACT_DIR / "metrics.json", "w") as f:
            json.dump(m, f, indent=2)
        print(f"metrics -> {ARTIFACT_DIR / 'metrics.json'}  (shap_summary.png in figures/)")
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    main(n=args.n, seed=args.seed, save=not args.no_save)
