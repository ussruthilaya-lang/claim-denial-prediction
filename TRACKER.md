# Deliverable Tracker

CS6140 Summer 2026 — **2-week project window: 2026-07-15 → 2026-07-29**
(final presentation day). Updated as work happens, not chronologically
archived. Status values: `todo` · `in-progress` · `blocked` · `done`.

Ownership matches [README.md](README.md): each phase owner does the model
**and** that phase's evaluation rigor (cost-sensitive metrics, calibration,
error analysis, label-risk notes) — rigor isn't a separate centralized job.

Two hard sequencing constraints given the compressed timeline:
1. **MIMIC-IV-Note CITI access (Phase 3, Het) must start 2026-07-15, day one** — approval can take days and blocks everything else in Phase 3.
2. **Phase 4's retrieval depends on Phases 1–3 having comparable features** — everyone posts to `shared/` early, not at the end.

## Prototype status (2026-07-15)

Nainica ran a real end-to-end prototype through all 4 phases in Colab —
`claim_denial_pipeline_3-3.ipynb` (repo root, untracked). This isn't a
sketch: it's actual code with actual metrics logged to local MLflow. It
proves the whole pipeline architecture works, end to end, in under two
weeks — that's the biggest risk-reduction event so far in the project.
Results:

| Model | AUROC | F1 | Notes |
|---|---|---|---|
| B1 — Logistic Regression (structured) | 0.5147 | 0.2055 | SMOTE + RFE, 5-fold CV |
| B2 — Decision Tree (structured) | 0.4992 | 0.1949 | SMOTE, 5-fold CV |
| B3 — Gradient Boosting (structured) | 0.5146 | 0.2905 | sklearn GB, not XGBoost — see note below |
| B3b — Random Forest (structured) | 0.5128 | 0.3106 | SHAP computed for both B3 and B3b |
| B4 — Hybrid (structured + ClinicalBERT) | 0.4902 | 0.2650 | On MTSamples proxy notes — see note below |
| B5 — Full RAG (structured + BERT + retrieval) | **0.5330** | **0.4820** | Best of all 5; P=0.3455, R=0.7976 |

Two deliberate deviations from the proposal, decided 2026-07-15:

1. **Phase 2 used sklearn GradientBoosting/RandomForest, not XGBoost/LightGBM.**
   Marked **done** as-is — GB/RF + SHAP is validated and sufficient for the
   ablation study. Swapping to XGBoost/LightGBM specifically is optional
   polish, only if time remains (see Week 2).
2. **Phases 3 and 4 both ran on MTSamples** (public clinical notes, matched
   to Kaggle claims by diagnosis-category bucket) **instead of MIMIC-IV-Note
   and Synthea+fault-injection.** The prototype's own comments flag this as
   a placeholder. Marked **done as proof-of-concept** — the pipeline logic
   (embed → fuse → retrain, index → retrieve → inject → retrain) is proven.
   The remaining work is swapping in the real data sources, tracked
   explicitly below, not re-deriving the pipeline.

Not yet done by anyone, any phase: porting this logic out of the notebook
into the owned package structure (`phase*_*/src/`) so it reads/writes
through `shared/schemas` and `shared/utils/eval.py`; GPT-4 zero-shot
baseline; cost-sensitive metrics; calibration; error analysis; label-risk
writeup.

## Phase 4 status (2026-07-22) — delivered, real data swap complete

Phase 4 (Sruthilaya) is **done and reproducible**, ported out of the prototype
into `phase4_rag_agentic/src/` and wired through `shared/schemas` +
`shared/utils/eval.py`. Run it with `python scripts/run_phase4.py`; interactive
demo is `streamlit run mlops_platform/demo/app.py`; narrative in
`phase4_pipeline.ipynb`.

**Key decision — decoupled from MIMIC.** MIMIC-IV-Note access never landed, but
Phase 4 never depended on it (it was always Synthea + injected labels). The
MTSamples placeholder is retired. Instead of the Synthea jar we use a **controlled
synthetic generator** (`data_gen.py`) with a documented, *learnable*
fault-injection denial rule (`labeling.py`) auto-calibrated to ~19% (CMS ACA
2024). Owning the rule is what lets us answer both instructor-feedback points as
experiments rather than caveats.

Ablation (40k claims, temporal split, leakage-safe), all through the shared eval:

| Model | AUROC | F1 | Note |
|---|---|---|---|
| Structured XGBoost | 0.742 | 0.417 | beats the prototype's 0.53 decisively — labels are learnable |
| + Retrieval features | **0.766** | **0.448** | real lift; recovers latent per-provider denial propensity |
| Oracle ceiling | 0.869 | — | max recoverable given the injected rule |
| Leaky index (INVALID) | 0.973 | 0.829 | what you'd wrongly report if the index contained the test claims |

Rigor delivered: leakage-safe temporal split + self-exclusion (with the leaky
variant reported as a cautionary contrast); calibration curve + Brier; cost-
sensitive operating point (saves ~$0.66M vs do-nothing on the test slice);
SHAP (top feature is a retrieval feature); **label-noise sweep + recover-the-rule**
(feedback #1); **PSI/KS harmonization report** (feedback #2); label-risk stated
explicitly. faiss and LLM SDKs are optional (numpy retrieval backend + mock
rationale), so it runs on any machine with no external services.

## Week 1 (2026-07-15 → 2026-07-21) — real data, ported code, working models

| Deliverable | Owner | Due | Status | Notes |
|---|---|---|---|---|
| `shared/schemas/claim.py` + `shared/utils/eval.py` skeleton (AUROC/F1) | Everyone | 07-16 | in-progress | Unblocks all phases |
| MIMIC-IV-Note CITI training + access request submitted | Het | 07-15 | todo | Start day one — longest lead time in the project |
| Phase 1: port LR + DT + SMOTE + RFE + stratified k-fold into `phase1_baseline/src/` | Het | 07-18 | done (prototype) → port pending | Logic proven at AUROC 0.5147/0.4992 — port into package, wire through `shared/utils/eval.py` |
| Phase 2: port GB + RF + SHAP into `phase2_gbm_shap/src/` | Nainica | 07-19 | done (prototype) → port pending | AUROC 0.5146/0.5128; swap to XGBoost/LightGBM only if Week 2 has slack |
| Phase 4: fault-injected labels (~19% prevalence) via controlled generator | Sruthilaya | 07-18 | done | `data_gen.py`+`labeling.py`; auto-calibrated to 0.19; retired MTSamples/Synthea-jar |
| Phase 4: FAISS index over historical claims | Sruthilaya | 07-21 | done | `retriever.py` real; numpy fallback for no-faiss envs; self-exclusion leakage guard |
| Phase 3: proxy label construction from ICD-10/CPT (once MIMIC access lands) | Het | 07-21 | blocked | Blocked on CITI access; MTSamples version already proven in prototype |

## Week 2 (2026-07-22 → 2026-07-28) — real data swap-in, rigor, ablation, demo

| Deliverable | Owner | Due | Status | Notes |
|---|---|---|---|---|
| Phase 3: re-run embed → fuse → retrain on real MIMIC-IV-Note (replacing MTSamples) | Het | 07-24 | todo | Pipeline logic already proven at AUROC 0.4902 on MTSamples — this is a data swap, not a redesign |
| Phase 3: GPT-4 zero-shot baseline | Het | 07-25 | todo | |
| Phase 4: index → retrieve → inject retrieval features → retrain, leakage-safe | Sruthilaya | 07-24 | done | AUROC 0.742→0.766 (ceiling 0.869); leaky variant 0.973 reported as contrast |
| Phase 4: feature harmonization vs. Phases 1–3 | Sruthilaya | 07-25 | done | `harmonization.py` PSI/KS; demoed on temporal drift; ready to run vs Kaggle frame |
| (Optional, if time remains) Phase 2: swap sklearn GB/RF for XGBoost/LightGBM | Nainica | 07-25 | todo | Polish only — not required for the ablation study to work |
| Cost-sensitive operating point + calibration curve, all 4 phases | Phase owner each | 07-25 | Phase 4 done; others todo | Phase 4 uses FN=$400/FP=$40; same assumption should carry to other phases |
| Error analysis: what each phase fixes vs. the one before it | Phase owner each | 07-26 | Phase 4 done; others todo | Phase 4: retrieval recovers latent provider signal the flat model can't see; label-noise sweep quantifies the ceiling |
| Label-risk writeup (Phase 3 proxy labels, Phase 4 injected labels) | Het / Sruthilaya | 07-26 | Phase 4 done; Phase 3 todo | Phase 4 label risk quantified via noise sweep + recover-the-rule, not just asserted |
| Full ablation study across all 4 phases, on real data | Everyone | 07-27 | todo | Blocked on all 4 phases reporting through `shared/utils/eval.py`; Phase 4 already does |
| Demo/presentation site (`mlops_platform/`) | Sruthilaya | 07-28 | done | `streamlit run mlops_platform/demo/app.py` — scores a claim, shows similar past denials + rationale + action, and a Results tab with all Phase 4 figures |
| Final presentation | Everyone | 07-29 | todo | |

## Risk watchlist

- **MIMIC CITI access never landed** → Phase 3 is the exposed phase, not Phase 4. Phase 4 is fully decoupled (Synthea-style generated claims + injected labels) and is done; it no longer depends on any Phase-3 output. Phase 3 needs its own fallback (MTSamples prototype, or a public de-identified note set) — that is Het's call, out of Phase 4's scope.
- **331K note encoding is slow** → cache embeddings incrementally starting day one of the real Phase 3 run; the 2,226-note MTSamples run already took ~74 minutes, so budget accordingly for MIMIC's scale.
- **Feature harmonization (Phase 4) is the single biggest integration risk** — start it as soon as Phase 1–3 schemas stabilize, not after Phase 4's own modeling is done.
- **B4 (hybrid) scored *below* the structured-only baselines in the prototype** (0.4902 vs 0.5146) — worth understanding before the real-data run: likely the MTSamples-to-claim linkage (diagnosis-category bucket matching) is too coarse to carry signal, not that clinical text is uninformative. Real per-patient MIMIC notes should behave differently — call this out explicitly in the error analysis rather than let it read as "the proposal's hypothesis was wrong."
