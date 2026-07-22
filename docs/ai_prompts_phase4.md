# AI Prompts Used — Phase 4 (Sruthilaya)

Required by the grading rubric ("AI Prompts Used" is mandatory). This documents
how Claude (Anthropic, via Claude Code) was used while building Phase 4. Prompts
are paraphrased to the intent; the full transcript is available on request.

## Scope of AI assistance
AI was used for **code scaffolding, design discussion, debugging, and figure
generation**. All modeling decisions (leakage-safe evaluation, the injected-label
mechanism, the retrieval-feature design, the cost model) were reviewed and are
explained in the report in the author's own words. No text was copied from
external sources without citation.

## Prompts (chronological, paraphrased)

1. **Framing / strategy.** "Here is our project proposal and the instructor
   feedback. Our MIMIC access failed; Phase 4 is mine and I have ~2 days. Help
   me scope a Phase 4 that stands on its own without MIMIC and shows real ML
   depth." → Established that Phase 4 (Synthea + injected labels + FAISS
   retrieval) never depended on MIMIC, and that owning the label-injection rule
   lets us turn the instructor's two feedback points into experiments.

2. **Data generator.** "Write a controlled synthetic claim generator matching
   our ClaimRecord schema, with correlated fields (billed amount depends on
   procedure, prior-auth on procedure, filing delay on payer) rather than
   independent columns." → `data_gen.py`.

3. **Fault-injection labels.** "Implement a documented, *learnable* denial rule
   over real denial drivers (untimely filing, missing prior auth, medical
   necessity, out-of-network, billing anomaly), auto-calibrated by bisection to
   a target prevalence (~19%, CMS ACA 2024)." → `labeling.py`.

4. **Retriever without faiss.** "faiss-cpu has no wheel on Python 3.14. Add a
   numpy exact-cosine backend behind the existing FAISS class with identical
   query semantics, and add a self-exclusion leakage guard." → `retriever.py`.

5. **Retrieval features + leakage.** "Design retrieval features (neighbour denial
   rate, similarity-weighted denial, agreement) and make the pipeline
   leakage-safe: temporal split, index over train only, self-exclusion. Also
   build a deliberately leaky variant to demonstrate the inflation." →
   `features.py`, `pipeline.py`.

6. **Debugging the retrieval lift.** "The augmented model isn't beating the
   structured one — why?" → Diagnosis: the retrieval embedding equalled the
   classifier's features, so neighbour signal was redundant. Fix: add a latent
   per-provider denial propensity (invisible to the flat model) and make the
   retrieval embedding provider-aware, so retrieval recovers signal the flat
   model cannot see.

7. **Label + harmonization rigor.** "Implement the two instructor-feedback
   experiments: a label-noise sweep + recover-the-rule check, and a PSI/KS
   population-shift report." → `label_audit.py`, `harmonization.py`.

8. **Decision-support / agentic layer.** "Wire the agentic predictor into a
   decision-support function: probability from the model, rationale from an LLM
   with a deterministic mock fallback so it runs with no API key." → `llm_demo.py`.

9. **Figures + demo.** "Generate report figures in one consistent palette
   (ablation, calibration, cost curve, SHAP, noise sweep, harmonization) and a
   Streamlit app that scores a claim, shows similar past denials, and displays
   the results." → `plots.py`, `mlops_platform/demo/app.py`.

## Verification
Every module was smoke-tested; the retriever leakage guard and nearest-neighbour
contract are covered by `phase4_rag_agentic/tests/test_retriever.py`. Reported
metrics come from `scripts/run_phase4.py` and `phase4_rag_agentic/artifacts/metrics.json`.
