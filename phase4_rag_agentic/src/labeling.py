"""
Fault-injection denial labeling for Phase 4.

WHY THIS MODULE IS THE POINT OF PHASE 4'S DATA STORY:
The proposal calls for "denial labels injected via a fault-injection methodology
calibrated to CMS ACA 2024 denial statistics, targeting ~19% prevalence." That
one sentence hides two things a careful reviewer will ask about, and the
instructor's feedback asked about the first directly:

  1. "If the proxy/injected labels are noisy, everything downstream is shaky."
  2. "Are the labels actually LEARNABLE, or did the prototype inject noise?"
     (The prototype's Phase 4 AUROC of 0.5330 is almost exactly chance, which
      is the fingerprint of labels with little learnable structure.)

We answer both by making the injection rule an explicit, documented,
log-additive risk model over real denial drivers. Each driver below maps to a
named, citable denial reason (CARC/RARC-style): missing prior authorization,
medical-necessity mismatch, out-of-network, untimely filing, and billing
anomaly. Because the rule is explicit we can (a) report a clean CEILING on
recoverable AUROC, (b) sweep label noise and watch metrics degrade, and (c)
show a billing reviewer *why* a claim was labeled — which the demo app surfaces.

The intercept is auto-calibrated by bisection so the realized prevalence lands
on the target (default 0.19) regardless of the sampled feature mix — that is
what "calibrated to CMS statistics" concretely means here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Log-odds contribution of each denial driver. These are the "fault" weights.
# Signs and relative magnitudes reflect real US claim-adjudication behavior:
# untimely filing is a near-automatic denial; missing prior auth and
# medical-necessity failures are the two largest avoidable-denial categories.
DRIVER_WEIGHTS = {
    "untimely_filing": 3.2,        # submitted after the payer's filing window
    "missing_prior_auth": 2.6,     # procedure required prior auth, none obtained
    "medical_necessity": 2.0,      # diagnosis not covered for this procedure (code-based)
    "undocumented_necessity": 1.6, # necessity not JUSTIFIED in the note (text-only signal)
    "out_of_network": 1.4,         # provider out-of-network for the payer
    "billing_anomaly": 1.2,        # billed amount far above the procedure norm
}
# Small payer-specific residual risk (Medicaid/self-pay run higher denial rates).
PAYER_RESIDUAL = {
    "medicare": 0.0, "medicaid": 0.35, "private": -0.1,
    "self_pay": 0.5, "other": 0.1,
}

# Human-readable denial reason attached to a denied claim, chosen as the driver
# with the largest positive contribution. Used by the demo and error analysis.
REASON_TEXT = {
    "untimely_filing": "CO-29 Untimely filing: submitted after payer window",
    "missing_prior_auth": "CO-197 Precertification/authorization absent",
    "medical_necessity": "CO-50 Non-covered: not deemed medically necessary",
    "undocumented_necessity": "CO-50/N115 Medical necessity not established by documentation",
    "out_of_network": "CO-242 Services not provided by network provider",
    "billing_anomaly": "CO-45 Charge exceeds fee schedule / documentation",
    "baseline": "No dominant driver (baseline payer risk)",
}


@dataclass
class LabelingResult:
    frame: pd.DataFrame          # input frame + denied / true_denial_prob / reason_code
    intercept: float             # calibrated intercept
    realized_prevalence: float
    driver_prevalence: dict      # fraction of claims tripping each driver


def _driver_contributions(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-claim log-odds contribution from each named denial driver.

    Reads only latent columns produced by data_gen; returns a frame whose
    columns are the driver names so we can both sum them (the risk model) and
    argmax them (the dominant reason for a denied claim)."""
    late = (frame["_filing_days"] > frame["_filing_window"]).astype(float)
    no_pa = (frame["_prior_auth_required"] & ~frame["_prior_auth_obtained"]).astype(float)
    necessity = (~frame["_necessity_ok"]).astype(float)
    # Necessity documented in the note? Undocumented raises denial risk and is
    # visible ONLY in the clinical note, not in any structured billing field.
    undocumented = (~frame["_necessity_documented"]).astype(float)
    oon = (~frame["_in_network"]).astype(float)
    # Billing anomaly ramps up only once the amount is well above the norm.
    anomaly = np.clip(frame["_amount_ratio"] - 1.6, 0, None)

    return pd.DataFrame({
        "untimely_filing": DRIVER_WEIGHTS["untimely_filing"] * late,
        "missing_prior_auth": DRIVER_WEIGHTS["missing_prior_auth"] * no_pa,
        "medical_necessity": DRIVER_WEIGHTS["medical_necessity"] * necessity,
        "undocumented_necessity": DRIVER_WEIGHTS["undocumented_necessity"] * undocumented,
        "out_of_network": DRIVER_WEIGHTS["out_of_network"] * oon,
        "billing_anomaly": DRIVER_WEIGHTS["billing_anomaly"] * anomaly,
    }, index=frame.index)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _calibrate_intercept(base_logodds: np.ndarray, target: float,
                         tol: float = 1e-4) -> float:
    """Bisection: find intercept b so mean(sigmoid(base + b)) == target.
    This is the concrete meaning of 'calibrated to a target prevalence'."""
    lo, hi = -20.0, 20.0
    for _ in range(200):
        mid = (lo + hi) / 2
        prev = _sigmoid(base_logodds + mid).mean()
        if abs(prev - target) < tol:
            return mid
        if prev < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def label_claims(frame: pd.DataFrame, target_prevalence: float = 0.19,
                 label_noise: float = 0.0, seed: int = 42) -> LabelingResult:
    """Inject denial labels.

    Parameters
    ----------
    target_prevalence : realized denial rate to calibrate to (0.19 = CMS ACA '24).
    label_noise       : fraction of labels randomly flipped AFTER sampling, to
                        emulate proxy-label noise. 0.0 = clean oracle labels.
                        Swept by `label_audit.py` to quantify robustness.

    Returns the frame with three added columns:
      true_denial_prob : P(denied) under the clean rule (the recoverable signal)
      denied           : the (possibly noisy) injected label used for training
      reason_code      : dominant denial driver for denied claims (demo/analysis)
    """
    rng = np.random.default_rng(seed)
    contrib = _driver_contributions(frame)
    payer_resid = frame["insurance_type"].map(PAYER_RESIDUAL).to_numpy()
    # Systematic per-provider denial propensity: latent, high-cardinality, and
    # invisible to the flat classifier — the signal retrieval exists to recover.
    prov_resid = frame["_provider_propensity"].to_numpy()

    base = contrib.sum(axis=1).to_numpy() + payer_resid + prov_resid
    intercept = _calibrate_intercept(base, target_prevalence)
    prob = _sigmoid(base + intercept)

    clean_label = (rng.random(len(frame)) < prob).astype(int)
    if label_noise > 0:
        flip = rng.random(len(frame)) < label_noise
        noisy = np.where(flip, 1 - clean_label, clean_label)
    else:
        noisy = clean_label

    # Dominant reason among denied claims (argmax of positive contributions).
    dominant = contrib.idxmax(axis=1)
    any_driver = contrib.to_numpy().max(axis=1) > 0
    reason = np.where(any_driver, dominant, "baseline")

    out = frame.copy()
    out["true_denial_prob"] = prob
    out["denied"] = noisy
    out["reason_code"] = [REASON_TEXT[r] if d else None
                          for r, d in zip(reason, noisy)]

    driver_prev = {c: float((contrib[c] > 0).mean()) for c in contrib.columns}
    return LabelingResult(
        frame=out,
        intercept=float(intercept),
        realized_prevalence=float(noisy.mean()),
        driver_prevalence=driver_prev,
    )


if __name__ == "__main__":
    from phase4_rag_agentic.src.data_gen import generate_claims

    gen = generate_claims(n=40_000, seed=0)
    res = label_claims(gen.frame, target_prevalence=0.19, seed=0)
    print(f"realized prevalence: {res.realized_prevalence:.4f} (target 0.19)")
    print(f"calibrated intercept: {res.intercept:.3f}")
    print("driver trip rates:")
    for k, v in sorted(res.driver_prevalence.items(), key=lambda x: -x[1]):
        print(f"  {k:20s} {v:6.3f}")
    denied = res.frame[res.frame["denied"] == 1]
    print("\ntop denial reasons among denied claims:")
    print(denied["reason_code"].value_counts().head())
