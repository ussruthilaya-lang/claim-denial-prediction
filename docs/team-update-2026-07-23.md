# Team update — unified-data pivot (Het, 2026-07-23)

**Nothing is pushed yet — this is for your review before it lands.** Short version
of what I changed, why, and what each of you should sanity-check in your phase.
Full rationale + numbers: [`docs/unified_data_architecture.md`](unified_data_architecture.md).

## Why I changed anything

1. **MIMIC-IV got rejected** (we finished CITI, PhysioNet still denied us) — so real
   discharge summaries are off the table for this deadline.
2. **The old Kaggle set has no signal.** Every model scored AUROC ≈ 0.5 on it —
   that's *random*, not "structured features are weak." Our thesis was untestable.
3. **The four phases were on three unrelated datasets**, so the cross-phase ablation
   (the whole point of the project) wasn't apples-to-apples.

**The fix:** unify all four phases onto Phase 4's existing synthetic generator, and
extend it to emit a **clinical note per claim**. One dataset, one population, so the
ablation is finally valid — and it clears both of Prof. Toutiaee's feedback points
by construction (one population = no distribution mismatch; owned rule = auditable,
noise-swept labels).

**What did NOT change:** the proposal's scope, the 4 phases, the methods, the eval
harness, ~19% prevalence. Only the *data substrate* changed.

## What changed, by file

**Shared / generator (please skim):**
- `phase4_rag_agentic/src/data_gen.py` — added a synthetic `clinical_note` per claim
  + a new latent driver `_necessity_documented`, and a `generate_dataset()` wrapper
  all phases now call. Structured fields are byte-identical to before (I used a
  separate RNG for the note text).
- `phase4_rag_agentic/src/labeling.py` — added one driver, `undocumented_necessity`
  (weight 1.6, CO-50/N115), so the note carries a real denial signal.
- `shared/utils/eval.py` — added `bootstrap_auroc_lift()` (paired bootstrap CI +
  p-value on a lift). Purely additive.
- `phase4_rag_agentic/src/retriever.py` — silenced a benign numpy warning (`errstate`
  guard); **numbers unchanged.**

**New files (my phases + the runner):**
- `phase1_baseline/src/train.py` — Phase 1 (LR/DT + SMOTE + RFE + k-fold) ported onto
  the unified data.
- `phase2_gbm_shap/src/train.py` — Phase 2 (GBM/RF + SHAP) ported onto the unified data.
- `phase3_clinicalbert/src/{text_encoder,pipeline}.py` — Phase 3 (TF-IDF/ClinicalBERT
  fused with structured, empty-note ablation, cost, calibration).
- `phase3_clinicalbert/src/llm_baseline.py` — the "GPT-4 zero-shot" baseline, on Claude
  (gated on a key; no-ops without one).
- `scripts/run_all_phases.py` — the unified 4-way ablation table + figure.
- `docs/unified_data_architecture.md` — full design doc.

## 👉 What each of you should check

- **Sruthilaya (Phase 4):** I edited your `labeling.py` and `data_gen.py`. Your
  **retrieval lift is preserved** (+0.022, structured 0.733 → augmented 0.755) and
  your whole pipeline/rigor still runs — but the new driver shifted the absolute
  numbers a little and raised the oracle ceiling (0.869 → 0.883). Artifacts are
  refreshed. Please confirm you're OK with the `labeling.py` change (it's your file).
- **Nainica (Phase 2):** I added `phase2_gbm_shap/src/train.py` running your GBM/RF +
  SHAP on the unified data — it now shows **real signal** (GBM 0.734, RF 0.731) instead
  of the 0.5 on Kaggle, and SHAP attributes to filing/billed/CPT/payer as expected.
  Please review it and adjust to match how you'd have written it.

## The payoff — unified ablation (one dataset, temporal test)

| Model | AUROC |
|---|---|
| LLM zero-shot (Claude Sonnet 5, note only) | 0.606 |
| structured — Phase 1/2 | 0.733 |
| + text — Phase 3 | **0.769** (+0.036, 95% CI [0.026, 0.046], p<0.0005) |
| + retrieval — Phase 4 | 0.755 (+0.022) |
| + both | **0.795** (+0.063) |
| oracle ceiling | 0.883 |

Text and retrieval recover **different** latent signals and stack — that's the "codes
→ context" story, now measurable and statistically clear.

## How to run it

```bash
pip install -e .            # deps are declared per phase
python scripts/run_all_phases.py     # the unified ablation table + figure
```

Figures/metrics land in `artifacts/` and each `phase*/artifacts/`. Happy to walk
through any of it or change anything before we push — flag it here.
