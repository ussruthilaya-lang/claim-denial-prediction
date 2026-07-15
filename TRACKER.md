# Deliverable Tracker

CS6140 Summer 2026 — **2-week project window: 2026-07-15 → 2026-07-29**
(final presentation day). Updated as work happens, not chronologically
archived. Status values: `todo` · `in-progress` · `blocked` · `done`.

Ownership matches [README.md](README.md): each phase owner does the model
**and** that phase's evaluation rigor (cost-sensitive metrics, calibration,
error analysis, label-risk notes) — rigor isn't a separate centralized job.

Two hard sequencing constraints given the compressed timeline:
1. **MIMIC-IV-Note CITI access (Phase 3, Het) must start 2026-07-15, day one** — approval can take days and blocks everything else in Phase 3.
2. **Phase 3 depends on Nainica's Phase 2 classifier code**, and **Phase 4's retrieval depends on Phases 1–3 having comparable features** — everyone posts to `shared/` early, not at the end.

## Week 1 (2026-07-15 → 2026-07-21) — get every phase to a working model

| Deliverable | Owner | Due | Status | Notes |
|---|---|---|---|---|
| `shared/schemas/claim.py` + `shared/utils/eval.py` skeleton (AUROC/F1) | Everyone | 07-16 | in-progress | Unblocks all phases — first thing done, not last |
| MIMIC-IV-Note CITI training + access request submitted | Het | 07-15 | todo | Start day one regardless of Phase 1 status — longest lead time in the project |
| Phase 1: EDA, preprocessing, LR + DT, SMOTE, stratified k-fold | Het | 07-18 | todo | Notebook baseline already got AUROC ≈ 0.5 — port and confirm, don't re-derive from scratch |
| Phase 2: XGBoost + LightGBM + SHAP | Nainica | 07-19 | todo | Can start against Phase 1's preprocessed data as soon as it lands (~07-17) |
| Phase 4: Synthea pull + fault-injected labels (~19% prevalence) | Sruthilaya | 07-18 | todo | |
| Phase 4: FAISS index over historical claims | Sruthilaya | 07-21 | todo | `retriever.py` already stubbed |
| Phase 3: proxy label construction from ICD-10/CPT (once access lands) | Het | 07-21 | blocked | Blocked on CITI access |

## Week 2 (2026-07-22 → 2026-07-28) — rigor, integration, ablation, demo

| Deliverable | Owner | Due | Status | Notes |
|---|---|---|---|---|
| Phase 3: batch-encode + cache embeddings, fuse with structured features, retrain classifier | Het | 07-24 | todo | Depends on Nainica's Phase 2 classifier — coordinate handoff early in week |
| Phase 3: GPT-4 zero-shot baseline | Het | 07-25 | todo | |
| Phase 4: top-k retrieval injected as inference-time context | Sruthilaya | 07-24 | todo | `agentic_predictor.py` already stubbed |
| Phase 4: feature harmonization vs. Phases 1–3 | Sruthilaya | 07-25 | todo | Different patient populations — biggest integration risk in the project |
| Cost-sensitive operating point + calibration curve, all 4 phases | Phase owner each | 07-25 | todo | Same $/claim FN-vs-FP assumption used across phases for comparability |
| Error analysis: what each phase fixes vs. the one before it | Phase owner each | 07-26 | todo | Feeds directly into the ablation study — don't leave for the last day |
| Label-risk writeup (Phase 3 proxy labels, Phase 4 injected labels) | Het / Sruthilaya | 07-26 | todo | Stated explicitly in the report, not left for a reader to catch |
| Full ablation study across all 4 phases | Everyone | 07-27 | todo | Blocked on all 4 phases reporting through `shared/utils/eval.py` |
| Demo/presentation site (`mlops_platform/`) | Sruthilaya | 07-28 | todo | Scoped down from full MLOps: surfaces each phase's saved predictions, SHAP plots, and a retrieval example — built last, over finished results, purely to sell the idea in the presentation |
| Final presentation | Everyone | 07-29 | todo | |

## Risk watchlist

- **MIMIC CITI approval slips past 07-15** → Phase 3 timeline compresses further; Het should have Phase 1 buffer-ready to absorb a short delay.
- **331K note encoding is slow** → cache embeddings incrementally starting day one of Phase 3 work, don't batch it all at the end.
- **Feature harmonization (Phase 4) is the single biggest integration risk** — start it as soon as Phase 1–3 schemas stabilize, not after Phase 4's own modeling is done.
