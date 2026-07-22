"""
Phase 4 end-to-end pipeline: generate -> label -> split -> index -> featurize
-> train -> evaluate, with the leakage guards and rigor the phase is graded on.

THE ONE THING TO GET RIGHT HERE IS LEAKAGE. A retrieval-augmented classifier
is unusually easy to fool yourself with, because the retrieval index is built
from labelled history and each training row can trivially retrieve itself. We
defend against that on two axes and MEASURE the defence so it is not just a
claim:

  * TEMPORAL split — train on earlier submission dates, test on later ones.
    Retrieval at inference can only ever see the past, so evaluation must too.
    A random split would let the index contain claims from the test period.
  * SELF-EXCLUSION — a training claim is in the index; it must not count itself
    among its own neighbours. We train a deliberately LEAKY variant without this
    guard and report the inflated number, so the gap is visible in the ablation.

Everything routes through `shared.utils.eval.evaluate` so Phase 4's numbers are
directly comparable to Phases 1-3 in the cross-phase ablation table.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from phase4_rag_agentic.src.data_gen import generate_claims, row_to_claim
from phase4_rag_agentic.src.features import (RETRIEVAL_FEATURE_NAMES,
                                             RetrievalFeaturizer,
                                             StructuredEncoder)
from phase4_rag_agentic.src.labeling import label_claims
from phase4_rag_agentic.src.retriever import ClaimRetriever
from shared.utils.eval import evaluate

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"

# Cost-sensitive assumption, shared across phases (see README "Evaluation & rigor").
# A missed denial (FN) means no appeal is filed and revenue is written off; a
# false alarm (FP) costs a reviewer's time. They are not symmetric.
COST_FN = 400.0   # $ per missed denial
COST_FP = 40.0    # $ per unnecessary manual review


def _xgb() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=350, max_depth=5, learning_rate=0.08,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=2,
        eval_metric="logloss", tree_method="hist", n_jobs=-1,
        random_state=0,
    )


@dataclass
class Phase4Artifacts:
    encoder: StructuredEncoder
    retriever: ClaimRetriever
    featurizer: RetrievalFeaturizer
    model_structured: XGBClassifier
    model_augmented: XGBClassifier
    struct_names: list[str]
    aug_names: list[str]
    test_frame: pd.DataFrame           # includes prob_structured / prob_augmented
    metrics: dict = field(default_factory=dict)
    cost_operating_point: dict = field(default_factory=dict)


def _temporal_split(labeled: pd.DataFrame, test_frac: float = 0.25):
    """Earliest (1-test_frac) submission dates -> train; latest -> test."""
    order = labeled.sort_values("submission_date").index.to_numpy()
    cut = int(len(order) * (1 - test_frac))
    train = labeled.loc[order[:cut]].reset_index(drop=True)
    test = labeled.loc[order[cut:]].reset_index(drop=True)
    return train, test


def build_and_train(n: int = 40_000, seed: int = 42, k: int = 10,
                    label_noise: float = 0.0,
                    compute_leak: bool = True) -> Phase4Artifacts:
    gen = generate_claims(n=n, seed=seed)
    labeled = label_claims(gen.frame, target_prevalence=0.19,
                           label_noise=label_noise, seed=seed).frame
    train, test = _temporal_split(labeled)

    # ---- classifier features (Xs, no provider) vs retrieval embedding (Xe,
    #      provider-aware). Keeping them separate is what lets us attribute any
    #      lift to the retrieval layer rather than to a feature the flat model
    #      already had. ----
    enc = StructuredEncoder().fit(train)
    Xs_train, Xs_test = enc.transform(train), enc.transform(test)
    Xe_train, Xe_test = enc.transform_embedding(train), enc.transform_embedding(test)

    # ---- index over TRAINING claims only (retrieval can't see the future) ----
    retriever = ClaimRetriever(embedding_dim=Xe_train.shape[1])
    train_claims = [row_to_claim(r) for _, r in train.iterrows()]
    retriever.add(Xe_train, train_claims)
    feat = RetrievalFeaturizer(retriever, k=k)

    # ---- retrieval features: exclude-self on train (it IS in the index), none
    #      on test (it is not). ----
    Rtr_train = feat.transform(Xe_train, train, exclude_self=True)
    Rtr_test = feat.transform(Xe_test, test, exclude_self=False)

    y_train = train["denied"].to_numpy()
    y_test = test["denied"].to_numpy()

    Xa_train = np.hstack([Xs_train, Rtr_train])
    Xa_test = np.hstack([Xs_test, Rtr_test])

    # ---- models ----
    m_struct = _xgb().fit(Xs_train, y_train)
    m_aug = _xgb().fit(Xa_train, y_train)

    p_struct = m_struct.predict_proba(Xs_test)[:, 1]
    p_aug = m_aug.predict_proba(Xa_test)[:, 1]

    # ---- LEAKAGE DEMO: the classic "build the FAISS index over the WHOLE
    #      dataset" bug. The index contains train AND test, and self-exclusion is
    #      off everywhere, so every row retrieves ITSELF at similarity 1.0 with
    #      its own label. Trained AND evaluated under this leaky regime, the model
    #      learns to trust its self-neighbour and the test AUROC inflates toward
    #      1.0 — a number that would look great in a report and be completely
    #      fraudulent. Reporting it next to the honest number is the point. ----
    results = {
        "B_structured_xgb": evaluate(y_test, p_struct, "phase4", "structured-xgb"),
        "B_retrieval_augmented": evaluate(y_test, p_aug, "phase4", "retrieval-augmented"),
    }
    if compute_leak:
        leaky_retriever = ClaimRetriever(embedding_dim=Xe_train.shape[1])
        test_claims = [row_to_claim(r) for _, r in test.iterrows()]
        leaky_retriever.add(np.vstack([Xe_train, Xe_test]), train_claims + test_claims)
        leaky_feat = RetrievalFeaturizer(leaky_retriever, k=k)
        Rtr_train_leaky = leaky_feat.transform(Xe_train, train, exclude_self=False)
        Rtr_test_leaky = leaky_feat.transform(Xe_test, test, exclude_self=False)
        m_leaky = _xgb().fit(np.hstack([Xs_train, Rtr_train_leaky]), y_train)
        p_leaky = m_leaky.predict_proba(np.hstack([Xs_test, Rtr_test_leaky]))[:, 1]
        results["B_leaky_index_has_test"] = evaluate(
            y_test, p_leaky, "phase4", "leaky-index")
    # Oracle ceiling: how much signal is recoverable at all, given the injected
    # rule. AUROC of the clean generative probability against the noisy labels.
    oracle_auroc = float(evaluate(
        y_test, test["true_denial_prob"].to_numpy(), "phase4", "oracle").auroc)

    metrics = {name: r.as_dict() for name, r in results.items()}
    metrics["oracle_ceiling_auroc"] = oracle_auroc
    metrics["backend"] = retriever.backend
    metrics["n_train"] = int(len(train))
    metrics["n_test"] = int(len(test))
    metrics["k"] = k
    metrics["label_noise"] = label_noise
    metrics["test_prevalence"] = float(y_test.mean())

    test = test.copy()
    test["prob_structured"] = p_struct
    test["prob_augmented"] = p_aug

    art = Phase4Artifacts(
        encoder=enc, retriever=retriever, featurizer=feat,
        model_structured=m_struct, model_augmented=m_aug,
        struct_names=enc.feature_names,
        aug_names=enc.feature_names + RETRIEVAL_FEATURE_NAMES,
        test_frame=test, metrics=metrics,
    )
    art.cost_operating_point = cost_optimal_threshold(y_test, p_aug)
    art.metrics["cost_operating_point"] = art.cost_operating_point
    return art


def cost_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Threshold minimizing expected $ cost = COST_FN*FN + COST_FP*FP.
    This is the operating point a real appeals team would run at, not 0.5."""
    thresholds = np.linspace(0.02, 0.98, 97)
    best = None
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        fn = int(((pred == 0) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        cost = COST_FN * fn + COST_FP * fp
        if best is None or cost < best["expected_cost"]:
            tp = int(((pred == 1) & (y_true == 1)).sum())
            best = {"threshold": float(t), "expected_cost": float(cost),
                    "fn": fn, "fp": fp, "tp": tp,
                    "cost_per_claim": float(cost / len(y_true))}
    # Baseline: flag nothing (file no appeals) — the do-nothing cost to beat.
    do_nothing = COST_FN * int((y_true == 1).sum())
    best["do_nothing_cost"] = float(do_nothing)
    best["savings_vs_do_nothing"] = float(do_nothing - best["expected_cost"])
    return best


def save_artifacts(art: Phase4Artifacts, out_dir: Path = ARTIFACT_DIR) -> None:
    import pickle
    out_dir.mkdir(parents=True, exist_ok=True)
    art.retriever.save(out_dir / "claims_index.faiss")
    with open(out_dir / "bundle.pkl", "wb") as f:
        pickle.dump({
            "encoder": art.encoder,
            "model_structured": art.model_structured,
            "model_augmented": art.model_augmented,
            "struct_names": art.struct_names,
            "aug_names": art.aug_names,
            "k": art.featurizer.k,
            "cost_operating_point": art.cost_operating_point,
        }, f)
    art.test_frame.to_parquet(out_dir / "test_predictions.parquet", index=False)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(art.metrics, f, indent=2)


if __name__ == "__main__":
    art = build_and_train()
    save_artifacts(art)
    m = art.metrics
    print(f"backend={m['backend']}  n_train={m['n_train']}  n_test={m['n_test']}")
    print(f"test prevalence={m['test_prevalence']:.3f}\n")
    print(f"{'model':28s} {'AUROC':>7s} {'F1':>7s} {'P':>7s} {'R':>7s}")
    for name in ["B_structured_xgb", "B_retrieval_augmented", "B_leaky_index_has_test"]:
        if name not in m:
            continue
        r = m[name]
        print(f"{name:28s} {r['auroc']:7.4f} {r['f1']:7.4f} "
              f"{r['precision']:7.4f} {r['recall']:7.4f}")
    print(f"\noracle ceiling AUROC: {m['oracle_ceiling_auroc']:.4f}")
    cop = m["cost_operating_point"]
    print(f"cost-optimal threshold={cop['threshold']:.2f}  "
          f"${cop['cost_per_claim']:.2f}/claim  "
          f"savings vs do-nothing=${cop['savings_vs_do_nothing']:,.0f}")
