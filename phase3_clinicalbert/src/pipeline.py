"""
Phase 3 pipeline: does the clinical NOTE add denial signal the structured
billing fields cannot? Runs the ablation that answers it, on the unified
generated dataset (same rows every phase sees), leakage-safe.

Design choices that keep it honest and comparable:
  * Same STRUCTURED features, temporal split, XGBoost config, and cost function
    as Phase 4 (imported, not re-implemented) — so `structured-only` here is the
    same model family as Phases 1/2/4 and the AUROCs line up in the ablation.
  * The note-only signal (`_necessity_documented`) is invisible to the structured
    fields by construction (see data_gen.py), so any lift is attributable to text.
  * EMPTY-NOTE ABLATION: blank every note and re-run. If the "lift" survives, it
    was leaking through the structured columns — the guardrail that keeps this
    from becoming a hidden version of Phase 4's leaky-index pathology.
  * Bounded by the ORACLE CEILING (the stochastic label caps recoverable AUROC
    well below 1.0), so a healthy result is a small lift, never an extreme number.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Allow `python phase3_clinicalbert/src/pipeline.py` from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sklearn.metrics import brier_score_loss  # noqa: E402

from phase3_clinicalbert.src.text_encoder import get_text_encoder  # noqa: E402
from phase4_rag_agentic.src.data_gen import generate_dataset  # noqa: E402
from phase4_rag_agentic.src.features import StructuredEncoder  # noqa: E402
from phase4_rag_agentic.src.pipeline import (_temporal_split, _xgb,  # noqa: E402
                                             cost_optimal_threshold)
from shared.utils.eval import bootstrap_auroc_lift, evaluate  # noqa: E402

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
PHASE = "phase3"


def run_phase3(n: int = 40_000, seed: int = 42, encoder: str = "tfidf",
               label_noise: float = 0.0) -> dict:
    df = generate_dataset(n=n, seed=seed, label_noise=label_noise)
    train, test = _temporal_split(df)
    y_tr, y_te = train["denied"].to_numpy(), test["denied"].to_numpy()

    # ---- structured features (identical to Phase 1/2/4 -> comparable) ----
    senc = StructuredEncoder().fit(train)
    Xs_tr, Xs_te = senc.transform(train), senc.transform(test)

    # ---- text features (fit vocabulary on TRAIN notes only) ----
    tenc = get_text_encoder(encoder).fit(train["clinical_note"].tolist())
    Xt_tr = tenc.transform(train["clinical_note"].tolist())
    Xt_te = tenc.transform(test["clinical_note"].tolist())

    Xf_tr = np.hstack([Xs_tr, Xt_tr]).astype("float32")
    Xf_te = np.hstack([Xs_te, Xt_te]).astype("float32")

    # ---- models: structured-only, text-only, fused ----
    p_struct = _xgb().fit(Xs_tr, y_tr).predict_proba(Xs_te)[:, 1]
    p_text = _xgb().fit(Xt_tr, y_tr).predict_proba(Xt_te)[:, 1]
    p_fused = _xgb().fit(Xf_tr, y_tr).predict_proba(Xf_te)[:, 1]

    r_struct = evaluate(y_te, p_struct, PHASE, "structured-only")
    r_text = evaluate(y_te, p_text, PHASE, f"text-only-{tenc.name}")
    r_fused = evaluate(y_te, p_fused, PHASE, f"structured+text-{tenc.name}")
    oracle = float(evaluate(y_te, test["true_denial_prob"].to_numpy(),
                            PHASE, "oracle").auroc)

    # ---- empty-note ablation: same channel, no content -> lift must vanish ----
    blank_tr = tenc.transform([""] * len(train))
    blank_te = tenc.transform([""] * len(test))
    p_blank = _xgb().fit(
        np.hstack([Xs_tr, blank_tr]).astype("float32"), y_tr
    ).predict_proba(np.hstack([Xs_te, blank_te]).astype("float32"))[:, 1]
    r_blank = evaluate(y_te, p_blank, PHASE, "structured+blank-note")

    # Is the text lift real or noise? Paired bootstrap CI + one-sided p-value.
    boot = bootstrap_auroc_lift(y_te, p_struct, p_fused, n_boot=2000, seed=seed)

    metrics = {
        "phase": PHASE,
        "encoder": tenc.name,
        "text_dim": int(tenc.dim) if getattr(tenc, "dim", None) else None,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "test_prevalence": float(y_te.mean()),
        "label_noise": label_noise,
        "structured_only": r_struct.as_dict(),
        "text_only": r_text.as_dict(),
        "structured_plus_text": r_fused.as_dict(),
        "structured_plus_blank_note": r_blank.as_dict(),
        "oracle_ceiling_auroc": oracle,
        "text_lift_auroc": float(r_fused.auroc - r_struct.auroc),
        "text_lift_bootstrap": boot,
        "empty_note_lift_auroc": float(r_blank.auroc - r_struct.auroc),
        "brier_structured": float(brier_score_loss(y_te, p_struct)),
        "brier_fused": float(brier_score_loss(y_te, p_fused)),
        "cost_operating_point": cost_optimal_threshold(y_te, p_fused),
    }
    return metrics


def _print_summary(m: dict) -> None:
    print("\n================ PHASE 3 SUMMARY ================")
    print(f"encoder={m['encoder']} (dim={m['text_dim']})  "
          f"n_train={m['n_train']} n_test={m['n_test']}  "
          f"test prevalence={m['test_prevalence']:.3f}")
    print(f"{'model':30s} {'AUROC':>7s} {'F1':>7s} {'P':>7s} {'R':>7s}")
    for key in ["structured_only", "text_only", "structured_plus_text",
                "structured_plus_blank_note"]:
        r = m[key]
        print(f"{key:30s} {r['auroc']:7.4f} {r['f1']:7.4f} "
              f"{r['precision']:7.4f} {r['recall']:7.4f}")
    print(f"{'oracle_ceiling':30s} {m['oracle_ceiling_auroc']:7.4f}")
    b = m["text_lift_bootstrap"]
    p = b["p_value_one_sided"]
    pstr = "<0.0005" if p == 0 else f"{p:.4f}"
    print(f"\ntext lift (fused - structured):   {m['text_lift_auroc']:+.4f}  "
          f"95% CI [{b['lift_ci'][0]:+.4f}, {b['lift_ci'][1]:+.4f}]  p(1-sided)={pstr}")
    print(f"empty-note lift (guardrail ~0):   {m['empty_note_lift_auroc']:+.4f}")
    print(f"Brier: structured={m['brier_structured']:.4f}  fused={m['brier_fused']:.4f}")
    cop = m["cost_operating_point"]
    print(f"cost-optimal: t={cop['threshold']:.2f}  ${cop['cost_per_claim']:.2f}/claim  "
          f"saves ${cop['savings_vs_do_nothing']:,.0f} vs do-nothing")

    # Healthy = positive text lift, below the ceiling, empty-note lift ~0.
    lift, empt = m["text_lift_auroc"], m["empty_note_lift_auroc"]
    ceil = m["oracle_ceiling_auroc"]
    fused = m["structured_plus_text"]["auroc"]
    flags = []
    if lift <= 0.005:
        flags.append("SIREN: no text lift — note signal not landing")
    if fused > ceil + 0.01:
        flags.append("SIREN: fused above oracle ceiling — leak")
    if abs(empt) > 0.01:
        flags.append("SIREN: empty-note lift non-zero — signal leaking via structured cols")
    if b["lift_ci"][0] <= 0:
        flags.append("WARN: lift 95% CI includes 0 — not statistically clear")
    print("verdict:", "  ".join(flags) if flags else
          "healthy — text lift bounded, statistically clear, guardrail clean")


def main(n: int = 40_000, seed: int = 42, encoder: str = "tfidf",
         save: bool = True) -> dict:
    m = run_phase3(n=n, seed=seed, encoder=encoder)
    _print_summary(m)
    if save:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        with open(ARTIFACT_DIR / "metrics.json", "w") as f:
            json.dump(m, f, indent=2)
        print(f"\nmetrics -> {ARTIFACT_DIR / 'metrics.json'}")
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--encoder", default="tfidf", choices=["tfidf", "clinicalbert"])
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    main(n=args.n, seed=args.seed, encoder=args.encoder, save=not args.no_save)
