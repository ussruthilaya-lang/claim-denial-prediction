# Unified Data Architecture (Path A) — Phase 3 Pivot Plan

**Status:** proposed · **Date:** 2026-07-22 · **Deadline:** 2026-07-29
**Owners:** Het (Phase 1 + Phase 3), Nainica (Phase 2), Sruthilaya (Phase 4 + demo)
**Decision this doc asks for:** approve pivoting Phase 3 (and the shared data
substrate for Phases 1–2) onto Phase 4's controlled generator, extended to emit
a clinical note per claim — *before* any code is written, and *before* pushing
to GitHub.

> This is a plan, not a code change. Nothing in `src/` changes until the team
> approves it and we've seen the scores from a trial run.

---

## 1. TL;DR

1. **MIMIC-IV credentialing was rejected** (CITI completed, PhysioNet denied).
   Real discharge summaries are off the table for this deadline.
2. **The old data substrate can't test our thesis.** Phases 1–3 ran on a 1,000-row
   Kaggle set whose denial label is essentially random — every model scores AUROC
   ≈ 0.5. That's *no signal*, not "structured features are insufficient." And the
   four phases live on three unrelated datasets, so the ablation isn't
   apples-to-apples.
3. **Fix: one controlled generator feeds all four phases.** Phase 4 already owns a
   realistic generator (`data_gen.py`) + an explicit, CMS-calibrated denial rule
   (`labeling.py`). We extend it to also emit a `clinical_note` per claim.
4. **The note carries a signal the billing fields do not** — documentation-based
   medical necessity — so ClinicalBERT (Phase 3) can recover something the
   structured model genuinely cannot see. That *is* the project thesis, made
   measurable.
5. **This clears both of Prof. Toutiaee's doubts architecturally**, not with
   caveats: one population by construction (doubt #2 dissolved), and an owned,
   auditable, noise-swept label with a gold hand-check (doubt #1 answered).

---

## 2. What stays the same (the core idea is untouched)

Path A changes **only the data substrate**. Every scientific commitment in the
proposal survives verbatim:

| Proposal commitment | Under Path A |
|---|---|
| "From Codes to Context" — does clinical narrative add signal over structured billing fields? | Unchanged — now testable on one linked dataset |
| Four progressive phases (structured → GBM+SHAP → ClinicalBERT → retrieval) | Unchanged |
| Phase 1: LR + DT, SMOTE, backward elimination, stratified k-fold | Unchanged |
| Phase 2: gradient boosting + SHAP per claim | Unchanged |
| Phase 3: ClinicalBERT embeddings fused with structured features; GPT-4 zero-shot baseline | Unchanged — runs on generated notes instead of MIMIC |
| Phase 4: FAISS retrieval over historical claims | Unchanged (already done) |
| Ablation isolating each component's contribution | Unchanged — and finally valid |
| Cost-sensitive metrics, calibration, error analysis, label-risk | Unchanged |
| ~19% denial prevalence, CMS ACA 2024 | Unchanged |

**What changes:** the three mismatched datasets (Kaggle claims + MIMIC notes +
Synthea) collapse into **one controlled generator** that emits structured fields,
a clinical note, and a denial label for the *same* synthetic patient. That is the
only change.

---

## 3. Why we must change the substrate (the evidence)

**Problem A — the Kaggle set has no learnable signal.** Prototype results, 5-fold CV:

| Model | AUROC | Reading |
|---|---|---|
| LR (structured) | 0.5147 | chance |
| Decision Tree | 0.4992 | chance |
| Gradient Boosting | 0.5146 | chance |
| Random Forest | 0.5128 | chance |
| Hybrid + ClinicalBERT | **0.4902** | *below* chance |
| Full RAG | 0.5330 | chance |

AUROC 0.49–0.53 across every model means the label is independent of the features.
The proposal's "confirms structured features are insufficient" is a misread: 0.5 is
*no signal at all*. The hybrid scored below baseline because notes were matched to
claims by a 3-way category bucket and **averaged**, giving only 3 distinct
embedding vectors across 1,000 claims — a 3-level categorical, not clinical text.

**Problem B — the phases aren't comparable.** Phase 1/2 (Kaggle), Phase 3 (Kaggle
claims + unrelated MTSamples notes), Phase 4 (separate synthetic world) sit on
different populations. Comparing their AUROCs — the whole point of an ablation —
is invalid. (This is precisely doubt #2.)

**Problem C — MIMIC is gone.** Rejected, not delayed. And even with access, MIMIC
has no denial field, so the note↔denial link was always going to be synthetic.
Losing MIMIC removes real *text*, not a capability we actually had.

---

## 4. The unified data architecture

### 4.1 One generator, one population, all four phases

```
                       ┌───────────────────────────────────────────┐
                       │  data_gen.generate_dataset(n, seed)         │
                       │  → one DataFrame per synthetic patient:     │
                       │     • observable structured fields          │
                       │     • clinical_note   (NEW)                 │
                       │     • latent drivers  (_-prefixed)          │
                       │     • denied / true_denial_prob / reason    │
                       └───────────────────────────────────────────┘
                                          │  ClaimRecord (shared/schemas)
        ┌─────────────────┬───────────────┼────────────────┬──────────────────┐
        ▼                 ▼               ▼                ▼                  ▼
   Phase 1           Phase 2         Phase 3           Phase 4          shared/utils/eval
   LR + DT           GBM + SHAP      ClinicalBERT      FAISS retrieval  (one metrics table)
   structured        structured      struct + NOTE     struct + history
```

Every phase reads the **same rows** and reports through `shared/utils/eval.py`, so
the ablation is finally apples-to-apples by construction.

### 4.2 The three-signal design (the heart of the plan)

The claim's denial probability is driven by three *kinds* of signal, each visible
to exactly one tier of the pipeline. This is what makes the progressive ablation
tell a clean story instead of four disconnected experiments:

| Signal | Lives in | Visible to | Recovered by | Real-world analogue |
|---|---|---|---|---|
| Filing timeliness, upcoding, code-based necessity, payer | observable billing fields | structured model | **Phase 1/2** | what a biller sees on the claim form |
| **Documentation-based medical necessity** (NEW) | the clinical note only | *not* the structured fields | **Phase 3 (ClinicalBERT)** | payer denies CO-50 because the note doesn't justify the procedure |
| Latent per-provider denial propensity | claim history only (`provider_id` withheld from the classifier) | *not* a single claim | **Phase 4 (retrieval)** | some practices simply have worse clean-claim discipline |

Each tier recovers a signal the previous tier structurally cannot see, so each
adds a **distinct, bounded** lift. Phase 4's provider-propensity story already
exists in the code (`features.py`, `PROVIDER_WEIGHT`, `provider_id` excluded from
`transform`). Path A adds the **symmetric Phase 3 story** on the text side.

### 4.3 The new latent driver: documentation-based medical necessity

The existing `_necessity_ok` (ICD/CPT compatibility) is **partly visible** to the
structured model, because the classifier already one-hot-encodes `cpt_code` and
`dx_category`. If we based Phase 3 on that, the tree could reconstruct it and
Phase 3 would show little lift (or worse, look like it's re-reading a structured
feature). So Path A introduces a **new** latent driver that no billing field can
express:

- `_necessity_documented` (bool): did the physician's note actually document
  justification for the procedure — e.g., failed conservative therapy, symptom
  severity, relevant comorbidities?
- It is realistic: a claim can have perfectly compatible codes (structured model
  sees nothing wrong) yet be denied for medical necessity because the *narrative*
  doesn't support it. That is a genuine CO-50 mechanism.
- It contributes to the denial log-odds in `labeling.py` (so the label depends on
  it), and it is written into `clinical_note` (so ClinicalBERT can read it), but
  it is **never** added to the observable/structured feature set.
- Weak, believable correlation with `_provider_quality` (better practices document
  better) is fine and realistic — but it must **not** be reconstructable from
  `cpt_code`, `dx_category`, `billed_amount`, `filing_days`, or `insurance_type`.
  (We verify this: see the empty-note ablation in §7.)

### 4.4 Note-generation spec

For each claim, `data_gen` composes a short synthetic clinical note from three
parts:

1. **Context (non-signal):** age/sex, diagnosis in words, procedure in words,
   generic history/exam filler. Gives ClinicalBERT realistic text to encode so
   the signal is not a single keyword.
2. **The signal (lossy, never the label):** if `_necessity_documented`, include
   justification language ("conservative management with NSAIDs and PT failed over
   6 weeks; imaging supports intervention"); if not, omit/undercut it
   ("procedure performed; prior conservative therapy not documented").
3. **Variation:** 3–5 templates per branch + synonym swaps, so the signal is
   *expressed*, not stamped. Lossy prose is what keeps the AUROC realistic instead
   of extreme (§4.5).

**Hard rules (the guardrails, spelled out):**
- The note **never** mentions the outcome, the word "denied/approved", or the
  `reason_code`. It describes the clinical/administrative situation only.
- The note is built from `_necessity_documented` (a driver), **never** from
  `denied` (the label). Building from the label is the leak that produced Phase
  4's cautionary 0.973 index.
- No verbatim numeric features ("filing_days=47"); use prose the model must
  interpret.
- GPT-4 zero-shot baseline (per the proposal) reads the same note and predicts
  denial — kept as a Phase 3 deliverable.

### 4.5 Why an extreme / "unexplained" AUROC is impossible here

The label is a **weighted coin flip**: `denied ~ Bernoulli(sigmoid(base + b))`.
Even a model that knows the exact true probability cannot perfectly predict it —
which is why Phase 4's **oracle ceiling is 0.869, not 1.0**. Every feature-based
model, text included, is bounded by that ceiling. Encoding the *driver* (not the
label) into the note caps Phase 3 at the ceiling, not at 1.0. An AUROC of 0.95+
is not reachable by this route — if it ever appeared, it would mean outcome text
leaked, and we'd know exactly where to look. The stochastic label is the
structural safety net.

### 4.6 Data flow / schema

`generate_dataset()` returns observable columns + `clinical_note` + latent drivers
+ label. Observable rows map 1:1 to `shared.schemas.claim.ClaimRecord` (add an
optional `clinical_note_text` field — the schema already anticipates
`clinical_note_text` for Phase 3). Latent `_`-columns are consumed only by
`labeling.py` and the audit, never by any model — the generator already enforces
this separation (`GeneratedClaims.observable_columns` vs `latent_columns`).

---

## 5. How the architecture clears the professor's two doubts

> "Since MIMIC-IV-Note doesn't have real denial labels, stress-test your proxy
> labeling method against a small hand-checked sample early on — if the proxy
> labels are noisy, everything downstream could be shaky." — doubt #1
>
> "Double-check that combining synthetic claims with real clinical notes from
> different patient populations doesn't introduce distribution mismatches that
> could quietly hurt later-phase results." — doubt #2

### 5.1 Doubt #1 — label trustworthiness

The pivot changes what this doubt even refers to: **there are no ICD/CPT proxy
labels under Path A.** The label comes from an explicit, owned rule. We answer the
*spirit* of the doubt — "can we trust the labels?" — four ways, three of which
already exist:

1. **Owned, documented, reality-grounded rule.** Every driver in `labeling.py`
   maps to a named CARC/RARC denial code (CO-29, CO-197, CO-50, CO-242, CO-45) and
   the prevalence is auto-calibrated to CMS ACA 2024 (~19%). Realism is *stated
   and cited*, not assumed. *(exists)*
2. **Noise sweep = robustness bound.** `label_audit.noise_sweep` flips 0→40% of
   labels and reports the AUROC decay toward chance — "how shaky if labels are
   noisy," made numeric. *(exists — but see the honesty note below)*
3. **Recover-the-rule.** `label_audit.recover_the_rule` shows the model learns the
   injected mechanism (Spearman rank vs true probability; each driver raises
   predicted risk). *(exists)*
4. **NEW — gold hand-audit (the honest analogue of his "hand-checked sample").**
   Because we own generation, we can render ~20 claims (note + structured, label
   hidden), have a human assign a denial reason, and compare to the injected
   reason. This is literally the hand-check he asked for, now *possible* because
   the substrate is controlled. *(to add — cheap, do it early)*

**Honesty note we will state out loud, not bury:** the noise sweep measures
*sensitivity to* noise, not the *amount* of noise in our labels — those are
different questions, and we won't present one as the other. And we cannot validate
against real denials, because no public dataset has real denial labels linked to
claims. That limitation is the finding, stated plainly (§8).

### 5.2 Doubt #2 — population mismatch

This doubt is **dissolved by construction**: the note and the claim are the *same
synthetic patient*, generated in one pass from one distribution. There is no
second population to mismatch. Compare the two paths we considered:

| | Path A (this plan) | Path B (real open notes + synthetic claims) |
|---|---|---|
| Note & claim population | identical (one DGP) | different (his exact warning) |
| Mismatch risk | none by construction | high |
| Can `harmonization.py` even measure the gap? | n/a | **no** — real notes share no columns with claims |

We still *run* `harmonization.population_shift_report` (PSI/KS) at the real
integration seam — the unified generated frame vs the Phase 1/2 frame — so
alignment is a **demonstrated, passed check**, not an assumption. (Path B, by
contrast, walks straight into doubt #2 *and* our tooling couldn't even detect it,
since real notes have none of the structured columns PSI compares.)

**Net:** doubt #2 is an argument *for* Path A. Doubt #1 is answered more cleanly by
an owned rule than it ever could have been by an un-auditable proxy.

---

## 6. Guardrails checklist (leakage · overfitting · honesty)

- [ ] Notes generated from **drivers**, never from `denied` / `reason_code`.
- [ ] The note-only driver (`_necessity_documented`) is **excluded** from all
      structured/observable features.
- [ ] Lossy prose, template variety — no verbatim feature values, no single-token
      tells.
- [ ] **Empty-note ablation:** strip the text → Phase 3 lift must vanish. If it
      doesn't, the signal is leaking through structured columns (bug).
- [ ] **Noise sweep** extended to the text lift: it must decay like Phase 4's.
- [ ] Reuse Phase 4's **leakage-safe temporal split** + retrieval self-exclusion.
- [ ] Report **train and test** AUROC; a large gap = overfitting, reported not hidden.
- [ ] Compare Phase 3 to the **oracle ceiling**; Phase 3 ≈ ceiling means the text
      is too on-the-nose — make it more oblique / add noise.
- [ ] Keep the **leaky-index variant** reported as the cautionary INVALID contrast.

---

## 7. Results (measured 2026-07-22, n=40k, seed=42)

Ran and confirmed — these are actual numbers, not projections. Every projection
in the earlier draft landed in its predicted band.

Classical baselines (`phase1_baseline/`, `phase2_gbm_shap/`), temporal test:

| Model | AUROC | Note |
|---|---|---|
| Phase 1 — LogReg (SMOTE+RFE) | 0.716 | real signal; was 0.51 on Kaggle. 5-fold CV 0.656±0.007 |
| Phase 1 — Decision Tree | 0.668 | |
| Phase 2 — Gradient Boosting (+SHAP) | 0.734 | SHAP: filing_days, log_billed, amount_ratio, cpt, payer |
| Phase 2 — Random Forest | 0.731 | |

Unified ablation (`scripts/run_all_phases.py`), lifts with paired-bootstrap 95% CI:

| Model | AUROC | Lift vs structured (95% CI) |
|---|---|---|
| Structured (XGBoost) | 0.733 | — |
| + note, **TF-IDF** (Phase 3) | **0.769** | +0.036 [0.026, 0.046], p<5e-4 |
| + note, ClinicalBERT (Phase 3) | 0.755 | +0.022 [0.011, 0.033], p<5e-4 |
| + retrieval (Phase 4) | 0.755 | +0.022 [0.013, 0.030], p<5e-4 |
| **+ both (note + retrieval)** | **0.795** | **+0.063** [0.052, 0.073], p<5e-4 |
| Oracle ceiling | 0.883 | — |

**Verdict: healthy on every check.** Monotonic climb; all lifts statistically
clear (CIs exclude 0); text and retrieval are complementary (+both recovers ~63%
of the gap to the ceiling); everything sits below the 0.883 ceiling — no extreme
number. Guardrails: **empty-note ablation lift = +0.0009** (the lift is note
content, not leakage); Brier improves 0.151→0.144.

**Honest finding:** frozen ClinicalBERT CLS embeddings *underperform* TF-IDF on
the templated notes (+0.022 vs +0.036) — expected without fine-tuning, and worth
reporting as a result in its own right rather than hiding. Fine-tuning (or a
better pooling/probe) is future work.

---

## 8. Honest limitations (stated, not discovered by a reader)

- The data is **synthetic**; results demonstrate the *method and the pipeline*,
  not a real-world denial rate. Framed exactly as Phase 4 already frames itself.
- We **cannot** validate labels against real denials — no public dataset has real
  denial labels linked to claims. That absence is itself a core finding of the
  project and the reason the field leans on proxy/injected labels.
- The note-driven lift is *by construction*; its honesty rests on the guardrails
  in §6 (driver-not-label, empty-note ablation, noise sweep, oracle ceiling).

---

## 9. Work plan (2026-07-22 → 07-29)

Sequenced so the label foundation is validated **first** (the professor's "early
on"), before anything is built on top.

| # | Task | Owner | Depends on |
|---|---|---|---|
| 0 | Approve this doc; email Prof. Toutiaee re: MIMIC rejection + re-mapped doubts | team / Het | — |
| 1 | Add `_necessity_documented` driver + note generator to `data_gen.py`/`labeling.py` | Sruthilaya + Het | 0 |
| 2 | Gold hand-audit (~20 claims) + face-validity write-up | Het | 1 |
| 3 | Port Phase 1 (LR/DT/SMOTE/RFE/k-fold) → `phase1_baseline/src/`, via shared eval | Het | 1 |
| 4 | Port Phase 2 (GBM/RF + SHAP) → `phase2_gbm_shap/src/`, via shared eval | Nainica | 1 |
| 5 | Phase 3: ClinicalBERT on notes + fuse + GPT-4 zero-shot → `phase3_clinicalbert/src/` | Het | 1,4 |
| 6 | Empty-note ablation + text noise sweep (guardrail proof) | Het | 5 |
| 7 | Cost op-point + calibration + error analysis for Phases 1–3 (copy Phase 4's pattern) | each owner | 3,4,5 |
| 8 | Unified 4-phase ablation table through `shared/utils/eval.py` | everyone | 3–6 |
| 9 | Run harmonization PSI/KS at the integration seam | Sruthilaya | 8 |
| 10 | **Trial run — review scores against §7 before pushing to GitHub** | team | 8 |
| 11 | Final presentation | everyone | 10 |

Two of the substantive pieces (Phase 1 port, Phase 3 build) are Het's; Phase 4 is
already done, so the generator extension is a small, contained change to code that
already works.

## 10. Future work (if the deadline weren't a week away)

- Re-appeal MIMIC with the professor as the **credentialed** supervising reference
  (a common rejection cause); institutional email; specific research description.
- With real MIMIC-IV (hosp + Notes), construct ICD/CPT proxy denial labels and
  repeat Phase 3 on genuine discharge summaries — the original vision, as a
  real-data validation of the synthetic result.

---

## Appendix — file-by-file change map (plan only, no code yet)

| File | Change | Kind |
|---|---|---|
| `phase4_rag_agentic/src/data_gen.py` | add `_necessity_documented`; add `clinical_note`; add `generate_dataset()` wrapper | edit |
| `phase4_rag_agentic/src/labeling.py` | add `undocumented_necessity` driver + CARC/RARC reason text | edit |
| `shared/schemas/claim.py` | ensure `clinical_note_text` optional field is present | edit |
| `phase1_baseline/src/{preprocess,train}.py` | port from notebook; read generated frame; report via shared eval | new |
| `phase2_gbm_shap/src/{train,shap_analysis}.py` | port from notebook; report via shared eval | new |
| `phase3_clinicalbert/src/{encode,fuse_features,train,gpt4_zeroshot}.py` | ClinicalBERT on notes + fuse + GPT-4 baseline | new |
| `phase4_rag_agentic/src/label_audit.py` | add `gold_hand_audit()` | edit |
| `scripts/run_all_phases.py` | one entry point → unified ablation table | new |
| `TRACKER.md` | update Phase 3 row: MIMIC rejected → unified-generator pivot | edit |
