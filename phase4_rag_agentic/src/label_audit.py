"""
Label-quality audit for Phase 4 — the direct answer to the instructor's
feedback #1: "stress-test your proxy/injected labels; if they are noisy,
everything downstream is shaky."

With real denial data you cannot do this, because you never see the true label.
Because Phase 4 OWNS the injection rule, it can, and does, three ways:

  1. recover_the_rule  — does the trained model actually learn the injected
     mechanism? We check rank agreement between predicted risk and the latent
     true probability, and confirm each named driver raises predicted risk.
  2. noise_sweep       — flip a growing fraction of labels and watch AUROC decay
     toward chance. This quantifies exactly how much label noise the pipeline
     tolerates before it stops working, which is the "how shaky?" question made
     numeric.
  3. prevalence_check  — confirm the auto-calibration hits the target denial
     rate across a range of targets (the "calibrated to CMS statistics" claim,
     verified rather than asserted).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from phase4_rag_agentic.src.data_gen import generate_claims
from phase4_rag_agentic.src.labeling import label_claims
from phase4_rag_agentic.src.pipeline import build_and_train


def recover_the_rule(art) -> dict:
    """Evidence that the model learned the injected rule, not noise."""
    test = art.test_frame
    p = test["prob_augmented"].to_numpy()
    truth = test["true_denial_prob"].to_numpy()
    rank_agreement = float(spearmanr(p, truth).statistic)

    # Each named denial driver should, on average, raise the model's predicted
    # risk. We recompute driver flags from the latent columns carried on the frame.
    late = test["_filing_days"] > test["_filing_window"]
    no_pa = test["_prior_auth_required"] & ~test["_prior_auth_obtained"]
    necessity = ~test["_necessity_ok"]
    oon = ~test["_in_network"]
    drivers = {"untimely_filing": late, "missing_prior_auth": no_pa,
               "medical_necessity": necessity, "out_of_network": oon}
    lift = {name: (float(p[mask].mean()), float(p[~mask].mean()))
            for name, mask in drivers.items() if mask.any()}
    return {"rank_agreement_spearman": rank_agreement, "driver_risk_lift": lift}


def noise_sweep(noise_levels=(0.0, 0.05, 0.10, 0.20, 0.30, 0.40),
                n: int = 25_000, seed: int = 7) -> pd.DataFrame:
    """Train the augmented model under increasing label noise; report the decay.
    Uses a lighter config (smaller n, no leakage variant) so the whole sweep
    runs in a couple of minutes."""
    rows = []
    for noise in noise_levels:
        art = build_and_train(n=n, seed=seed, label_noise=noise, compute_leak=False)
        m = art.metrics
        rows.append({
            "label_noise": noise,
            "auroc_structured": m["B_structured_xgb"]["auroc"],
            "auroc_augmented": m["B_retrieval_augmented"]["auroc"],
            "f1_augmented": m["B_retrieval_augmented"]["f1"],
            "oracle_ceiling": m["oracle_ceiling_auroc"],
        })
    return pd.DataFrame(rows)


def prevalence_check(targets=(0.10, 0.15, 0.19, 0.25, 0.30),
                     n: int = 30_000, seed: int = 3) -> pd.DataFrame:
    """Confirm the intercept auto-calibration lands on the requested prevalence."""
    gen = generate_claims(n=n, seed=seed)
    rows = []
    for t in targets:
        res = label_claims(gen.frame, target_prevalence=t, seed=seed)
        rows.append({"target_prevalence": t,
                     "realized_prevalence": res.realized_prevalence,
                     "abs_error": abs(res.realized_prevalence - t)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("== prevalence calibration ==")
    print(prevalence_check().to_string(index=False))
    print("\n== label-noise sweep (this trains several models) ==")
    print(noise_sweep().to_string(index=False))
