# From Codes to Context: Claim Denial Prediction

CS6140 · Khoury College · Northeastern University · Summer 2026

Nainica Dasari · Het Suryakant Prajapati · Sruthilaya Umasankari Soma Shanmuga Sundaram

## Problem

Medical claim denials cost U.S. hospitals ~$19.7B/year. Existing automated
denial prediction relies only on structured billing fields (ICD-10, CPT,
insurance type, submission intervals) and ignores clinical narrative in
physician notes. This project asks whether unstructured clinical text and
historical denial patterns improve denial prediction over structured-only
baselines.

## Scope

The full plan is four progressive phases (below), each independently
evaluable, each producing AUROC/F1 comparable across phases in a final
ablation study. **All four phases are in scope for the course deliverable** —
there is no reduced "core" set, as promised in the proposal.

What's simplified is *when infra gets built*, not the modeling scope:
- No Docker/Terraform/CI/MLflow up front — that was solving a problem we
  don't have yet.
- The one platform piece that stays in scope is a **presentation/demo site**
  (owned by Sruthilaya, built last): a lightweight app that pulls each
  phase's saved predictions, SHAP output, and retrieval examples into one
  place to visually sell the idea at the final presentation. It's a demo
  layer over finished results, not production serving infra — see
  [TRACKER.md](TRACKER.md).

## Phases

Each phase owner is responsible for their model **and** the evaluation rigor
below (cost-sensitive metrics, calibration, error analysis) — see
[Evaluation & rigor](#evaluation--rigor).

| Phase | Data | Method | Owner |
|---|---|---|---|
| 1 — Structured baseline | Kaggle synthetic claims (davidcsullivan) | Reproduce Hiremath et al.: logistic regression + decision tree, SMOTE, backward elimination, stratified k-fold CV | Het |
| 2 — Gradient boosting + SHAP | Same as Phase 1 | XGBoost + LightGBM, SHAP attribution per claim | Nainica |
| 3 — Clinical text | MIMIC-IV-Note (PhysioNet), proxy denial labels via ICD-10/CPT | ClinicalBERT embeddings concatenated with structured features, retrained Phase 2 classifier; GPT-4 zero-shot baseline for comparison | Het |
| 4 — Retrieval-augmented | Synthea (MITRE), fault-injected denial labels (~19% prevalence, calibrated to CMS ACA 2024 stats) | FAISS index over historical claims; top-k similar past denials injected as context at inference | Sruthilaya |

Ownership is split so each person owns one "classical" and one "advanced"
piece where possible, and so no one person is stuck doing only baseline
work or only novel work — Phase 3 depends on Phase 2's classifier, so Het
(Phase 3) and Nainica (Phase 2) coordinate at that handoff. Sruthilaya's
Phase 4 plus the demo site is the platform-facing track, sized to match a
single phase plus a scoped-down (not full MLOps) presentation layer.

Preliminary result (Phase 1 sanity check, Kaggle notebook): logistic
regression and Random Forest on structured fields alone both plateau at
ROC AUC ≈ 0.5 — confirms structured-only features are insufficient and
motivates Phases 3–4.

## Evaluation & rigor

AUROC/F1 alone reads as a class exercise. To show industrial thinking
without building infra we don't need yet, every phase reports:

- **Cost-sensitive metrics** — a false negative (missed denial, no appeal
  filed) and a false positive (wasted appeal effort) aren't symmetric costs.
  Each phase reports a precision/recall operating point chosen against an
  assumed $-per-claim cost, not just AUROC.
- **Calibration** — a calibration curve (or Brier score) alongside
  discrimination metrics, since a threshold-based appeal workflow needs
  trustworthy probabilities, not just correct ranking.
- **Error analysis** — for phases 2–4, a short writeup of *which* claims
  the new phase fixes that the previous phase missed. This is the actual
  payoff of the progressive design and feeds the final ablation study.
- **Label risk, stated explicitly** — Phase 3's MIMIC labels are proxy
  labels (no ground-truth denial field); Phase 4's Synthea labels are
  fault-injected. Both are flagged in the phase's report, not discovered
  by a reader.

## Repo layout

```
.
├── phase1_baseline/       # Het — LR + DT, SMOTE, stratified k-fold
├── phase2_gbm_shap/       # Nainica — XGBoost/LightGBM + SHAP
├── phase3_clinicalbert/   # Het — ClinicalBERT + MIMIC-IV-Note
├── phase4_rag_agentic/    # Sruthilaya — FAISS retrieval + LLM-assisted inference
├── shared/                # Everyone — claim schema, eval harness (AUROC/F1, cost-sensitive metrics, calibration)
├── mlops_platform/demo/   # Sruthilaya, built last — Streamlit app over finished phase results (`make demo`)
├── data/                  # raw/ processed/ external — gitignored
├── docs/adr/              # architecture decisions, kept with their reasoning
└── scripts/               # one-off setup / data-download scripts
```

Every phase reads from `shared/schemas` and writes metrics through
`shared/utils/eval.py` so results stay comparable across phases.

## Quickstart

```bash
pip install -e .
pytest
```

The demo site quickstart will be documented once `mlops_platform/` has
results to show — see [TRACKER.md](TRACKER.md).

## Tracking

Deliverables, owners, and status live in [TRACKER.md](TRACKER.md), not here —
keep this README as the stable project description and let the tracker
change week to week.
