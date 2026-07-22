"""
One-shot Phase 4 runner: trains the models, runs the label audit and the
harmonization check, writes every figure and metrics.json under
phase4_rag_agentic/artifacts/, and prints the headline table.

    python scripts/run_phase4.py            # full run (includes noise sweep)
    python scripts/run_phase4.py --fast     # skip the multi-model noise sweep
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python scripts/run_phase4.py` from anywhere by putting the repo root
# (this file's parent's parent) on the import path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase4_rag_agentic.src import plots
from phase4_rag_agentic.src.harmonization import population_shift_report
from phase4_rag_agentic.src.label_audit import (noise_sweep, prevalence_check,
                                                recover_the_rule)
from phase4_rag_agentic.src.pipeline import (ARTIFACT_DIR, build_and_train,
                                             save_artifacts)


def main(fast: bool = False) -> None:
    print("[1/6] training models (structured / retrieval-augmented / leaky)…")
    art = build_and_train(n=40_000, seed=42)
    save_artifacts(art)

    print("[2/6] figures: ablation, calibration, cost curve, SHAP…")
    plots.plot_ablation(art.metrics)
    plots.plot_calibration(art)
    plots.plot_cost_curve(art)
    plots.plot_shap(art)

    print("[3/6] label audit: recover-the-rule…")
    art.metrics["recover_the_rule"] = recover_the_rule(art)

    print("[4/6] label audit: prevalence calibration…")
    art.metrics["prevalence_check"] = prevalence_check().to_dict(orient="records")

    print("[5/6] harmonization: train vs test population shift…")
    from phase4_rag_agentic.src.data_gen import generate_claims
    from phase4_rag_agentic.src.labeling import label_claims
    from phase4_rag_agentic.src.pipeline import _temporal_split
    labeled = label_claims(generate_claims(n=40_000, seed=42).frame, seed=42).frame
    tr, te = _temporal_split(labeled)
    report = population_shift_report(tr, te)
    plots.plot_harmonization(report)
    art.metrics["harmonization"] = report.to_dict(orient="records")

    if not fast:
        print("[6/6] label-noise sweep (trains several models, ~2 min)…")
        sweep = noise_sweep()
        plots.plot_noise_sweep(sweep)
        art.metrics["noise_sweep"] = sweep.to_dict(orient="records")
    else:
        print("[6/6] skipped noise sweep (--fast)")

    with open(ARTIFACT_DIR / "metrics.json", "w") as f:
        json.dump(art.metrics, f, indent=2, default=str)

    m = art.metrics
    print("\n================ PHASE 4 SUMMARY ================")
    print(f"backend={m['backend']}  n_train={m['n_train']}  n_test={m['n_test']}  "
          f"test prevalence={m['test_prevalence']:.3f}")
    print(f"{'model':26s} {'AUROC':>7s} {'F1':>7s}")
    for key in ["B_structured_xgb", "B_retrieval_augmented", "B_leaky_index_has_test"]:
        if key in m:
            print(f"{key:26s} {m[key]['auroc']:7.4f} {m[key]['f1']:7.4f}")
    print(f"oracle ceiling AUROC: {m['oracle_ceiling_auroc']:.4f}")
    print(f"recover-the-rule Spearman: "
          f"{m['recover_the_rule']['rank_agreement_spearman']:.3f}")
    cop = m["cost_operating_point"]
    print(f"cost-optimal: t={cop['threshold']:.2f}  ${cop['cost_per_claim']:.2f}/claim  "
          f"saves ${cop['savings_vs_do_nothing']:,.0f} vs do-nothing")
    print(f"figures written to {plots.FIG_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    main(**vars(ap.parse_args()))
