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

## Phase 4 extension — requirement-level evidence RAG (2026-07-24, done — Sruthilaya's part; pending team merge into shared report)

Sruthilaya is extending Phase 4 with a second, deeper RAG: instead of
retrieving similar *claims*, this retrieves whether a specific LCD
*requirement* is satisfied by the patient's record. Scoped to 2 procedure
families (advanced imaging, PT/rehab), it targets the same 2 pieces of
instructor feedback that Phase 4's headline ablation already addresses —
proxy-label validation and real-vs-synthetic population harmonization —
applied one level deeper (requirement, not claim). Same retrieval rigor
(leakage-safe, cited rationale) — not a new deviation, an extension of the
existing one.

Verified 2026-07-24 by re-running both retrievers end-to-end against
`evidence_data_gen.py`'s 150 cases:

| Method | Accuracy | Notes |
|---|---|---|
| TF-IDF baseline (`evidence_retriever.py`) | 0.800 | Fails on paraphrased/indirect evidence (IMG-3/4/5) — zero lexical overlap |
| Semantic (`evidence_retriever_semantic.py`, all-MiniLM-L6-v2) | 0.827 | Fixes IMG-4/5 (paraphrase); new failures on IMG-2/3 (abstract requirement phrasing vs. concrete clinical text) |

Score separation (semantic, `rec_score`): complete/omitted mean **0.525** vs.
unsupported mean **0.258** — clean separation despite the IMG-2/3 failure
mode.

Done: `evidence_policies.py` (10 real CMS LCD-derived requirements, 2
families), `evidence_data_gen.py` (150 fault-injected cases, 50/50/50 split),
both retrievers above, `artifacts/ablation_summary.json`.

Gold-check subset (2026-07-24, done): hand-verified 18 of 150 cases (12%
sample, `random.Random(7)`) against the injected evidence templates and
`full_record_chunks`/`submitted_chunks` split. **18/18 (100%) agreement** —
`complete` cases have the evidence chunk in both submitted and full record,
`omitted` cases have it in the full record only, `unsupported` cases have no
evidence chunk anywhere and `gold_evidence_text` is correctly `None`. This is
the same "stress-test your proxy labels early" move the instructor asked for
at the claim level, applied here at the requirement level — the injection
logic does what it claims.

Population harmonization (feedback #2) — **scoped out, not missing.** The
evidence cases have no billing-style fields (`billed_amount`, `cpt_code`,
etc.) to run `harmonization.py`'s PSI/KS check against; that check is a
different unit of analysis (claim-level fields vs. requirement-level text
evidence). Feedback #2 is already answered at the claim level by Phase 4's
own PSI/KS report; the extension states this explicitly in the writeup
rather than forcing an artificial structured-field comparison.

Figures (2026-07-24, done) — `plots.py` extended with
`plot_evidence_confusion()` and `plot_evidence_score_distribution()`, same
palette/style as the rest of Phase 4:
- `evidence_confusion_tfidf.png`, `evidence_confusion_semantic.png` — gold
  vs. predicted evidence status. Semantic confusion matrix confirms the
  known failure cluster precisely: 10/50 `complete` and 10/50 `omitted`
  cases predicted as `unsupported` (the IMG-2/3 abstract-phrasing gap);
  `unsupported` itself is the most reliable class (46/50 correct).
- `evidence_score_distribution.png` — rec_score histogram, complete/omitted
  vs. unsupported. Clean bimodal separation (means 0.525 vs 0.258) with a
  visible overlap band ~0.2-0.3 — shown honestly, not smoothed over.

Cited deficiency report (2026-07-24, done) — `evidence_report.py`, mirroring
`llm_demo.py`'s exact pattern: status + cited chunk come from the
deterministic semantic retriever (never the LLM), a `mock_llm` composes the
reviewer-facing explanation with zero network dependency, and an optional
real LLM (Anthropic/OpenAI key) only upgrades the prose. Verified against
one case per gold variant — all three predicted correctly and produce a
concrete, actionable explanation (e.g., omitted case: "the PT progress note
... exists in the record but was not included in what was submitted ...
adding this note to the packet should resolve the deficiency"). Also fixed
a broken relative import in `evidence_data_gen.py` (`from src.evidence_policies`
→ `from phase4_rag_agentic.src.evidence_policies`) that would have silently
blocked this integration outside one specific working directory.

Demo app integration (2026-07-24, done) — new third tab "Evidence
completeness (extension)" in `mlops_platform/demo/app.py`, built for
transparency: a scope banner states up front that this is a separate,
narrower test from the Phase 1-4 denial model in the other two tabs (2
procedure families, 10 requirements, 150 hand-checked synthetic cases).
Every step is shown, not just the final answer — submitted vs. full-record
chunks side by side, both TF-IDF and semantic retrieval scores in one table,
the final status compared against the case's known injected ground truth
(labeled as such, since this is a scoped eval case, not a real claim), and
the accuracy/confusion-matrix/score-distribution results at the bottom.
Verified end-to-end with Streamlit's `AppTest` headless harness (no browser
tooling available in this environment) — no exceptions, and selecting the
IMG-4 "omitted" case correctly shows TF-IDF missing it (predicts
unsupported) while semantic catches it (predicts omitted), exactly matching
the documented failure/fix pattern.

Demo UX pass (2026-07-24, done) — reworked the extension tab from a
markdown-scribble layout into an equal-width 3-stage pipeline (Input →
Retrieve → Generate, fixed-height bordered cards), surfaced the actual
retrieved chunk (not just similarity scores) so the RAG retrieve-then-generate
flow is visible, and added a live cache-hit/computed indicator so the
"why is this instant" question has an honest, visible answer rather than
an unexplained fast demo. Also gave the claim-denial tab's claim record its
own visible input card (previously the chosen claim's fields were never
shown before scoring).

Data provenance (2026-07-24, done, stated here for the report): CMS LCD
requirement *text* is real, paraphrased from public CMS LCD L34220/L37281
(lumbar MRI) and L33942 (PT/rehab) policy documentation — not synthetic.
What's synthetic is the *evidence* side: the patient chart chunks (PT
progress notes, exam findings) that satisfy or fail to satisfy those real
requirements, and the case-assembly logic (which chunks go in the submitted
packet vs. the full record). Real Synthea patient bundles were not
downloadable in this build environment (no internet access to the Synthea
data generator/repository at build time), so the patient side is
schema-realistic-but-authored rather than literally downloaded — the same
scope decision Phase 4's base claim generator already made and disclosed,
applied one level deeper. This is stated directly, not left for a reader to
discover.

Both items originally on this list are done.

Report section (2026-07-24, done — Sruthilaya's part) —
`docs/report_phase4_section.md`: Methods/Dataset/Results/Discussion/Future
Work for Phase 4 + the evidence extension, plus an AI-prompts log scoped to
the intellectual decisions (why the extension was scoped this way, the
templated-vs-LLM data-generation call, why harmonization was explicitly
skipped rather than forced, the gold-check-before-building-on-top
discipline, demo transparency choices) rather than routine build steps.
Ready to merge into the shared 8-page report; Het/Nainica sections still
pending.

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
