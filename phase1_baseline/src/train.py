"""
Phase 1 — structured baseline (logistic regression + decision tree), reproducing
Hiremath et al.: SMOTE + RFE + stratified k-fold, now on the UNIFIED generated
dataset so the AUROC is directly comparable to Phases 2-4 in the ablation.

The point this phase makes: on the old Kaggle data every model sat at ~0.5 (no
signal); on the unified dataset the SAME classical methods reach a real
structured baseline. That baseline is the floor the later phases must beat.

Evaluated two ways, both through shared.utils.eval so they line up across phases:
  * temporal held-out test — the common protocol for the cross-phase ablation.
  * stratified 5-fold CV on train — the proposal's Phase-1 methodology / a
    robustness check that the single split isn't a fluke.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from imblearn.over_sampling import SMOTE  # noqa: E402
from imblearn.pipeline import Pipeline as ImbPipeline  # noqa: E402
from sklearn.feature_selection import RFE  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from phase4_rag_agentic.src.data_gen import generate_dataset  # noqa: E402
from phase4_rag_agentic.src.features import StructuredEncoder  # noqa: E402
from phase4_rag_agentic.src.pipeline import _temporal_split  # noqa: E402
from shared.utils.eval import evaluate  # noqa: E402

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
PHASE = "phase1"


def _lr_pipeline() -> ImbPipeline:
    """SMOTE -> RFE (backward elimination) -> logistic regression."""
    return ImbPipeline([
        ("smote", SMOTE(random_state=42)),
        ("rfe", RFE(LogisticRegression(solver="liblinear", max_iter=1000,
                                       random_state=42), n_features_to_select=15)),
        ("clf", LogisticRegression(solver="liblinear", max_iter=1000,
                                   random_state=42)),
    ])


def _dt_pipeline() -> ImbPipeline:
    return ImbPipeline([
        ("smote", SMOTE(random_state=42)),
        ("clf", DecisionTreeClassifier(max_depth=6, random_state=42)),
    ])


def run_phase1(n: int = 40_000, seed: int = 42) -> dict:
    df = generate_dataset(n=n, seed=seed)
    train, test = _temporal_split(df)
    enc = StructuredEncoder().fit(train)
    Xtr, Xte = enc.transform(train), enc.transform(test)
    ytr, yte = train["denied"].to_numpy(), test["denied"].to_numpy()

    results = {}
    for name, pipe in [("logreg-rfe-smote", _lr_pipeline()),
                       ("decision-tree-smote", _dt_pipeline())]:
        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xte)[:, 1]
        results[name] = evaluate(yte, p, PHASE, name).as_dict()

    # Stratified 5-fold CV on TRAIN (proposal methodology / robustness).
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv = [roc_auc_score(ytr[va], _lr_pipeline().fit(Xtr[tr], ytr[tr])
                        .predict_proba(Xtr[va])[:, 1])
          for tr, va in skf.split(Xtr, ytr)]
    cv = np.asarray(cv)

    return {
        "phase": PHASE,
        "n_train": int(len(train)), "n_test": int(len(test)),
        "test_prevalence": float(yte.mean()),
        "logreg": results["logreg-rfe-smote"],
        "decision_tree": results["decision-tree-smote"],
        "logreg_cv5_auroc_mean": float(cv.mean()),
        "logreg_cv5_auroc_std": float(cv.std()),
    }


def main(n: int = 40_000, seed: int = 42, save: bool = True) -> dict:
    m = run_phase1(n=n, seed=seed)
    print("\n================ PHASE 1 SUMMARY ================")
    print(f"n_train={m['n_train']} n_test={m['n_test']} prevalence={m['test_prevalence']:.3f}")
    print(f"{'model':22s} {'AUROC':>7s} {'F1':>7s} {'P':>7s} {'R':>7s}")
    for key in ["logreg", "decision_tree"]:
        r = m[key]
        print(f"{key:22s} {r['auroc']:7.4f} {r['f1']:7.4f} {r['precision']:7.4f} {r['recall']:7.4f}")
    print(f"LR stratified 5-fold CV AUROC: {m['logreg_cv5_auroc_mean']:.4f} "
          f"+/- {m['logreg_cv5_auroc_std']:.4f}")
    if save:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        with open(ARTIFACT_DIR / "metrics.json", "w") as f:
            json.dump(m, f, indent=2)
        print(f"metrics -> {ARTIFACT_DIR / 'metrics.json'}")
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    main(n=args.n, seed=args.seed, save=not args.no_save)
