# From Codes to Context: Claim Denial Prediction

CS6140 · Khoury College · Northeastern University · Summer 2026

Nainica Dasari · Het Suryakant Prajapati · Sruthilaya Umasankari Soma Shanmuga Sundaram

## Why this matters

U.S. hospitals lose an estimated $19.7B/year to claim denials. Most automated
denial-prediction tools look only at structured billing fields — ICD-10, CPT,
insurance type, filing dates — and never see the clinical narrative that
often explains *why* a payer actually denies a claim. Billing teams end up
either over-appealing (wasted effort) or under-appealing (lost revenue)
because the signal that would tell them which is which lives in a physician's
note, not a billing code.

This project asks a concrete question: **does clinical text and retrieval
over historical claims add real, measurable predictive lift over
structured-only baselines** — and can that lift be produced in a way that's
auditable, not just a bigger number?

## Approach

Four models are built progressively, each designed to recover a signal the
previous one structurally cannot see, so the lift at each stage tells a clean
causal story rather than four disconnected experiments:

| Stage | Recovers | Method |
|---|---|---|
| 1 — Structured baseline | Filing timeliness, coding mismatches, payer patterns | Logistic regression + decision tree, SMOTE, stratified k-fold |
| 2 — Gradient boosting + SHAP | Non-linear interactions between structured fields | XGBoost/GBM + SHAP attribution per claim |
| 3 — Clinical text | Documentation-based medical necessity — visible only in the note, not any billing field | ClinicalBERT + TF-IDF encodings, fused with structured features; GPT-4 zero-shot baseline |
| 4 — Retrieval-augmented | Latent per-provider denial propensity — visible only across claim history, not a single claim | Retrieval over historical claims, injected as features and as cited context at inference |
| 4-ext — Evidence RAG | Whether a specific payer requirement (real CMS LCD policy) is actually supported by the record, vs. omitted, vs. absent | TF-IDF + semantic (sentence-transformer) retrieval at the requirement level, scoped to 2 procedure families |

All four phases and the extension train on **one unified, controlled
synthetic generator** rather than four disconnected datasets — every phase
reads the same synthetic patient population, so the cross-phase ablation
comparison is apples-to-apples by construction, and every reported number
runs through one shared evaluation harness (`shared/utils/eval.py`). Full
architecture rationale: [docs/unified_data_architecture.md](docs/unified_data_architecture.md).

## Results

Unified ablation, temporal split, leakage-safe (`scripts/run_all_phases.py`):

| Model | AUROC | Lift vs. structured |
|---|---|---|
| Structured (XGBoost) | 0.733 | — |
| + clinical text (TF-IDF) | 0.769 | +0.036 |
| + clinical text (ClinicalBERT) | 0.755 | +0.022 |
| + retrieval | 0.755 | +0.022 |
| **+ text and retrieval combined** | **0.795** | **+0.063** |
| Oracle ceiling (max recoverable) | 0.883 | — |

The lift is monotonic, statistically clear (bootstrapped 95% CIs exclude
zero), and text/retrieval contribute complementary signal — combined, they
recover roughly two-thirds of the gap to the theoretical ceiling.

Evidence-level extension (150 cases, 2 CMS LCD-derived procedure families):
TF-IDF baseline retrieves the correct evidence status 80.0% of the time;
adding semantic retrieval (sentence-transformers) improves this to 82.7% by
recovering paraphrased evidence a keyword match misses. The two methods have
complementary strengths — lexical matching is more reliable on concrete,
literal phrasing, semantic matching on indirect phrasing — which is itself
a useful finding for where a production system would combine both.

Every phase reports beyond AUROC/F1: a cost-sensitive operating point (an
appealed-but-not-denied claim costs differently than a missed denial),
calibration, SHAP attribution, and an explicit statement of label risk —
covered in detail per-phase and tracked live in [TRACKER.md](TRACKER.md).

## Who built what

- **Het** — Phase 1 (structured baseline) and Phase 3 (clinical text
  fusion + GPT-4 zero-shot baseline); led the pivot to a unified data
  architecture after real MIMIC-IV-Note access fell through, which is what
  makes the four-phase comparison valid in the first place.
- **Nainica** — Phase 2 (gradient boosting + SHAP); ran the first full
  end-to-end prototype across all four phases early, which de-risked the
  whole project's architecture before the real-data build began.
- **Sruthilaya** — Phase 4 (retrieval-augmented prediction) and its
  requirement-level evidence-completeness extension, plus the demo/presentation
  layer that ties every phase's results into one place.

## Evaluation & rigor

Every phase reports, not just AUROC:

- **Cost-sensitive metrics** at a chosen precision/recall operating point,
  reflecting that a missed denial and a wasted appeal aren't symmetric costs.
- **Calibration** (curve + Brier score), since a threshold-based appeal
  workflow needs trustworthy probabilities, not just correct ranking.
- **Error analysis** — what each phase recovers that the previous one missed.
- **Label risk, stated explicitly** — every phase's labels are synthetic or
  proxy by necessity (no public dataset links real claims to real denial
  outcomes at scale); rather than treat that as a caveat, we validate it
  directly: a label-noise sweep quantifies how much the model's score decays
  as labels get noisier, a recover-the-rule check confirms the model learns
  the actual injected mechanism, and a hand-audited gold subset cross-checks
  the auto-generated labels against human judgment.
- **Population harmonization** — a PSI/KS drift report confirms the phases'
  populations are statistically comparable, since the unified generator is
  what makes that comparison meaningful in the first place.

## Repo layout

```
.
├── phase1_baseline/       # Het — LR + DT, SMOTE, stratified k-fold
├── phase2_gbm_shap/       # Nainica — gradient boosting + SHAP
├── phase3_clinicalbert/   # Het — ClinicalBERT + GPT-4 zero-shot baseline
├── phase4_rag_agentic/    # Sruthilaya — retrieval-augmented prediction + evidence-completeness RAG
├── shared/                # Everyone — claim schema, eval harness (AUROC/F1, cost-sensitive metrics, calibration)
├── mlops_platform/demo/   # Sruthilaya — Streamlit app presenting every phase's results (`make demo`)
├── docs/adr/              # architecture decisions, kept with their reasoning
└── scripts/               # data generation, per-phase runners, unified ablation
```

## Quickstart

```bash
pip install -e .
pytest

# generate the unified dataset + run every phase's ablation
python scripts/run_all_phases.py

# Phase 4 alone (retrieval-augmented prediction)
python scripts/run_phase4.py

# interactive demo — scores a claim, shows cited retrieval evidence, and
# a results tab with every phase's figures
streamlit run mlops_platform/demo/app.py
```

## Status

All four phases have a working, reproducible pipeline on real (unified, generated) data; 
the evidence-completeness extension and a few rigor items (gold-audit,
harmonization report re-run, final demo integration) are actively in
progress. Live status, owners, and dates: [TRACKER.md](TRACKER.md).
