"""
Unified cross-phase ablation on ONE linked dataset — the deliverable the whole
project is graded on, and the thing the old three-dataset design could never
produce validly.

Generates the dataset once, takes one leakage-safe temporal split, then builds
the four feature sets and trains the ladder:

    structured  ->  +text (Phase 3)  ->  +retrieval (Phase 4)  ->  +both

Everything shares the same StructuredEncoder, temporal split, XGBoost config, and
shared eval, so the AUROCs are strictly comparable. Retrieval features carry
Phase 4's leakage guards (index over TRAIN only; self-exclusion on train). Each
lift over the structured baseline gets a paired bootstrap CI + one-sided p-value,
and the oracle ceiling bounds what is recoverable at all.

    python scripts/run_all_phases.py                 # TF-IDF text (fast)
    python scripts/run_all_phases.py --encoder clinicalbert
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase3_clinicalbert.src.text_encoder import get_text_encoder  # noqa: E402
from phase4_rag_agentic.src.data_gen import (generate_dataset,  # noqa: E402
                                             row_to_claim)
from phase4_rag_agentic.src.features import (RetrievalFeaturizer,  # noqa: E402
                                             StructuredEncoder)
from phase4_rag_agentic.src.pipeline import (_temporal_split, _xgb,  # noqa: E402
                                             cost_optimal_threshold)
from phase4_rag_agentic.src.retriever import ClaimRetriever  # noqa: E402
from shared.utils.eval import bootstrap_auroc_lift, evaluate  # noqa: E402

ART_DIR = Path(__file__).resolve().parents[1] / "artifacts"
LADDER = ["structured", "structured+text", "structured+retrieval",
          "structured+text+retrieval"]


def run_ablation(n: int = 40_000, seed: int = 42, k: int = 10,
                 encoder: str = "tfidf") -> dict:
    df = generate_dataset(n=n, seed=seed)
    train, test = _temporal_split(df)
    ytr, yte = train["denied"].to_numpy(), test["denied"].to_numpy()

    # ---- structured features (shared by every phase) ----
    enc = StructuredEncoder().fit(train)
    Xs_tr, Xs_te = enc.transform(train), enc.transform(test)

    # ---- text features (fit vocabulary on train notes only) ----
    tenc = get_text_encoder(encoder).fit(train["clinical_note"].tolist())
    Xt_tr = tenc.transform(train["clinical_note"].tolist())
    Xt_te = tenc.transform(test["clinical_note"].tolist())

    # ---- retrieval features (Phase 4 leakage guards) ----
    Xe_tr, Xe_te = enc.transform_embedding(train), enc.transform_embedding(test)
    retr = ClaimRetriever(embedding_dim=Xe_tr.shape[1])
    retr.add(Xe_tr, [row_to_claim(r) for _, r in train.iterrows()])
    feat = RetrievalFeaturizer(retr, k=k)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")  # quiet numpy-backend similarity warnings
        Rtr_tr = feat.transform(Xe_tr, train, exclude_self=True)
        Rtr_te = feat.transform(Xe_te, test, exclude_self=False)

    feature_sets = {
        "structured": (Xs_tr, Xs_te),
        "structured+text": (np.hstack([Xs_tr, Xt_tr]), np.hstack([Xs_te, Xt_te])),
        "structured+retrieval": (np.hstack([Xs_tr, Rtr_tr]), np.hstack([Xs_te, Rtr_te])),
        "structured+text+retrieval": (np.hstack([Xs_tr, Xt_tr, Rtr_tr]),
                                      np.hstack([Xs_te, Xt_te, Rtr_te])),
    }

    probs, models = {}, {}
    for name, (Xtr, Xte) in feature_sets.items():
        p = _xgb().fit(Xtr.astype("float32"), ytr).predict_proba(Xte.astype("float32"))[:, 1]
        probs[name] = p
        models[name] = evaluate(yte, p, "ablation", name).as_dict()

    oracle = float(evaluate(yte, test["true_denial_prob"].to_numpy(),
                            "ablation", "oracle").auroc)
    base = probs["structured"]
    lifts = {name: bootstrap_auroc_lift(yte, base, probs[name], n_boot=2000, seed=seed)
             for name in LADDER[1:]}

    return {
        "encoder": tenc.name,
        "text_dim": int(tenc.dim) if getattr(tenc, "dim", None) else None,
        "n_train": int(len(train)), "n_test": int(len(test)),
        "k": k, "test_prevalence": float(yte.mean()),
        "models": models,
        "oracle_ceiling_auroc": oracle,
        "lifts": lifts,
        "cost_operating_point_full": cost_optimal_threshold(
            yte, probs["structured+text+retrieval"]),
    }


def _plot_ladder(m: dict, path: Path) -> None:
    aurocs = [m["models"][k]["auroc"] for k in LADDER]
    labels = ["structured\n(P1/P2)", "+text\n(P3)", "+retrieval\n(P4)", "+both"]
    colors = ["#9CA3AF", "#4C78A8", "#59A14F", "#B07AA1"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, aurocs, color=colors)
    ax.axhline(m["oracle_ceiling_auroc"], ls="--", color="#E15759",
               label=f"oracle ceiling {m['oracle_ceiling_auroc']:.3f}")
    ax.axhline(0.5, ls=":", color="#999999", label="chance 0.500")
    ax.set_ylim(0.45, 0.92)
    ax.set_ylabel("AUROC (temporal held-out test)")
    ax.set_title(f"Cross-phase ablation — unified dataset (encoder={m['encoder']})")
    for b, a in zip(bars, aurocs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.006, f"{a:.3f}",
                ha="center", fontsize=9)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(n: int = 40_000, seed: int = 42, k: int = 10, encoder: str = "tfidf",
         save: bool = True) -> dict:
    m = run_ablation(n=n, seed=seed, k=k, encoder=encoder)
    print("\n============== UNIFIED CROSS-PHASE ABLATION ==============")
    print(f"encoder={m['encoder']}  n_train={m['n_train']} n_test={m['n_test']}  "
          f"test prevalence={m['test_prevalence']:.3f}")
    print(f"{'model':30s} {'AUROC':>7s} {'F1':>7s}  {'lift vs structured (95% CI)':>28s}")
    for name in LADDER:
        r = m["models"][name]
        if name in m["lifts"]:
            b = m["lifts"][name]
            p = b["p_value_one_sided"]
            pstr = "<5e-4" if p == 0 else f"{p:.3f}"
            lift = f"{b['lift_point']:+.4f} [{b['lift_ci'][0]:+.3f},{b['lift_ci'][1]:+.3f}] p={pstr}"
        else:
            lift = "—  (baseline)"
        print(f"{name:30s} {r['auroc']:7.4f} {r['f1']:7.4f}  {lift:>28s}")
    print(f"{'oracle ceiling':30s} {m['oracle_ceiling_auroc']:7.4f}")
    cop = m["cost_operating_point_full"]
    print(f"\nfull-model cost-optimal: t={cop['threshold']:.2f}  "
          f"${cop['cost_per_claim']:.2f}/claim  "
          f"saves ${cop['savings_vs_do_nothing']:,.0f} vs do-nothing")
    if save:
        ART_DIR.mkdir(parents=True, exist_ok=True)
        suffix = "" if m["encoder"] == "tfidf" else f"_{m['encoder']}"
        with open(ART_DIR / f"unified_ablation{suffix}.json", "w") as f:
            json.dump(m, f, indent=2)
        _plot_ladder(m, ART_DIR / f"unified_ablation{suffix}.png")
        print(f"\nmetrics -> {ART_DIR / f'unified_ablation{suffix}.json'}")
        print(f"figure  -> {ART_DIR / f'unified_ablation{suffix}.png'}")
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--encoder", default="tfidf", choices=["tfidf", "clinicalbert"])
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    main(n=args.n, seed=args.seed, k=args.k, encoder=args.encoder, save=not args.no_save)
