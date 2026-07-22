"""
Feature harmonization / population-shift diagnostics — the direct answer to the
instructor's feedback #2: "combining synthetic claims with clinical notes from a
different patient population could introduce distribution mismatches that quietly
hurt later-phase results."

The cross-phase ablation only means something if the phases score comparable
populations. Phase 1/2 use Kaggle synthetic claims; Phase 4 uses Synthea-style
claims; a naive comparison of their AUROCs is invalid if their feature
distributions differ. This module is the reusable check that catches that,
using two standard drift metrics:

  * PSI (Population Stability Index) — the industry-standard drift score.
        PSI < 0.10 : negligible shift
        0.10-0.25  : moderate shift, investigate
        > 0.25     : major shift, populations not comparable as-is
  * KS statistic (+ p-value) for continuous fields.

We demonstrate it on Phase 4's OWN temporal train/test split — which genuinely
drifts, because late-filed (and therefore denial-prone) claims land in the later
test window — so the report shows a real, non-trivial finding rather than a
contrived one. `population_shift_report` takes any two claim frames, so the same
call harmonizes Phase 4 against the Phase 1/2 Kaggle frame at integration time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from phase4_rag_agentic.src.features import _ICD_TO_CAT, _filing_days


def psi_numeric(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """PSI over quantile bins of `expected`. Small epsilon guards empty bins."""
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_categorical(expected: pd.Series, actual: pd.Series) -> float:
    cats = sorted(set(expected.unique()) | set(actual.unique()))
    e = expected.value_counts(normalize=True).reindex(cats).fillna(0).to_numpy()
    a = actual.value_counts(normalize=True).reindex(cats).fillna(0).to_numpy()
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def _drift_flag(psi: float) -> str:
    return "major" if psi > 0.25 else ("moderate" if psi > 0.10 else "negligible")


def population_shift_report(ref: pd.DataFrame, cmp: pd.DataFrame) -> pd.DataFrame:
    """Per-feature PSI (+ KS for numerics) between two claim populations."""
    ref = ref.assign(_fd=_filing_days(ref),
                     _cat=ref["icd10_code"].map(_ICD_TO_CAT))
    cmp = cmp.assign(_fd=_filing_days(cmp),
                     _cat=cmp["icd10_code"].map(_ICD_TO_CAT))

    rows = []
    for name, col in [("billed_amount", "billed_amount"), ("filing_days", "_fd")]:
        psi = psi_numeric(ref[col].to_numpy(), cmp[col].to_numpy())
        ks = ks_2samp(ref[col], cmp[col])
        rows.append({"feature": name, "type": "numeric", "psi": psi,
                     "ks_stat": float(ks.statistic), "ks_pvalue": float(ks.pvalue),
                     "drift": _drift_flag(psi)})
    for name, col in [("insurance_type", "insurance_type"), ("cpt_code", "cpt_code"),
                      ("dx_category", "_cat")]:
        psi = psi_categorical(ref[col], cmp[col])
        rows.append({"feature": name, "type": "categorical", "psi": psi,
                     "ks_stat": np.nan, "ks_pvalue": np.nan,
                     "drift": _drift_flag(psi)})
    if "denied" in ref and "denied" in cmp:
        rows.append({"feature": "denial_prevalence", "type": "label",
                     "psi": np.nan, "ks_stat": np.nan, "ks_pvalue": np.nan,
                     "drift": f"{ref['denied'].mean():.3f} -> {cmp['denied'].mean():.3f}"})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from phase4_rag_agentic.src.data_gen import generate_claims
    from phase4_rag_agentic.src.labeling import label_claims
    from phase4_rag_agentic.src.pipeline import _temporal_split

    labeled = label_claims(generate_claims(n=40_000, seed=42).frame, seed=42).frame
    train, test = _temporal_split(labeled)
    print("Population shift: Phase 4 train (early) vs test (late)")
    print(population_shift_report(train, test).to_string(index=False))
