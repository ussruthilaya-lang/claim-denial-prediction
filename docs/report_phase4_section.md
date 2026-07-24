# Report section — Phase 4: Retrieval-Augmented Denial Prediction & Evidence-Completeness Extension

*(Sruthilaya's contribution — to be merged into the team's single 8-page report. Phases 1–3 sections owned by Het/Nainica.)*

## Methods

Phase 4 tests whether retrieval over historical claims adds predictive lift
beyond structured billing fields and clinical text alone. A claim is embedded
and matched against an indexed history of past claims (numpy or FAISS
backend); the retrieved neighborhood is converted into features (neighbor
denial rate, dominant denial reason, per-provider propensity) and appended to
the structured/text feature set before retraining. Retrieval is
**leakage-safe by construction**: the index is built only from the temporal
training split, and each query self-excludes its own claim, so no claim is
ever matched against itself or against future data. A **leaky-index
variant** (index includes test claims) is deliberately also reported, scored
at AUROC 0.973 vs. the valid 0.795 — presented as a cautionary contrast
showing what an invalid pipeline would look like, not a result.

A second, deeper extension applies the same retrieval philosophy one level
down: instead of asking "is this claim similar to past denials," it asks
"does this specific payer requirement have supporting evidence in this
patient's record, and was that evidence actually submitted?" Two retrievers
are compared: a TF-IDF baseline (global vectorizer, fit once, cosine
similarity) and a semantic retriever (`sentence-transformers/all-MiniLM-L6-v2`
embeddings, cosine similarity). Both output one of three states —
`complete` (evidence present and submitted), `omitted` (evidence exists in
the record but was withheld from the submission), `unsupported` (no
evidence exists anywhere) — plus a cited chunk and a reviewer-facing
rationale, mirroring the claim-level decision-support pattern used elsewhere
in Phase 4 (deterministic explanation by default; an optional LLM only
upgrades the prose, never the underlying retrieved status).

### RAG architecture: why retrieve-then-generate, not classify-then-explain

Both retrieval systems in Phase 4 are structured as retrieve → augment →
generate, and that separation is the actual methodological point, not an
implementation detail. In a standard classifier, the "explanation" for a
prediction (a SHAP value, an attention weight) is a property of the model's
internals — it can be technically correct and still not correspond to
anything a human could independently check. RAG inverts that: the system's
output is *grounded* in a specific, retrieved piece of evidence (a past
claim, a cited chart sentence) that exists independently of the model and
can be checked directly against the source. This is why the generation
step is deliberately kept "dumb" — a deterministic template over the
retrieved evidence by default, with an LLM only allowed to rephrase, never
to introduce or alter what was retrieved. If generation were allowed to
reason freely instead of being grounded in the retrieved chunk, the system
would be exposed to hallucination — a fluent-sounding explanation that
doesn't actually correspond to the record. Keeping retrieval and generation
strictly separated is what prevents that: the model cannot claim evidence
exists that the retriever did not actually find.

The choice between the two retrievers (TF-IDF vs. sentence-transformer
embeddings) is itself a lexical-vs-semantic tradeoff worth stating
explicitly, not just measuring: TF-IDF matches on shared vocabulary, so it
is precise but brittle to paraphrase (a requirement asking about
"radiculopathy" will not match a note that only says "numbness radiating
down the leg"); embedding-based retrieval matches on learned semantic
similarity, so it recovers paraphrased evidence at the cost of sometimes
conflating superficially different but conceptually related statements
(the IMG-2/3 failure mode). Reporting both, rather than only the better
one, is what makes this a methods comparison instead of a single number.

## Dataset & Inputs

All four phases and this extension train on **one unified, controlled
synthetic generator** (a team-wide pivot after MIMIC-IV-Note credentialing
was rejected), so every phase scores a comparable population and the
cross-phase ablation is valid by construction. Phase 4's claim-level data
(40k claims) includes structured billing fields, a synthesized clinical
note, and a denial label injected from an explicit, documented rule
calibrated to ~19% prevalence (CMS ACA 2024).

The evidence-completeness extension uses its own, smaller generator (150
cases: 50 each of complete/omitted/unsupported), built for a different unit
of analysis — a (requirement, patient-record) pair rather than a claim. Ten
requirement statements are paraphrased from two real CMS Local Coverage
Determination policies (L34220/L37281, lumbar MRI; L33942, PT/rehab
documentation standards) — this text is real and sourced, not generated.
The patient evidence chunks that satisfy or fail those requirements are
templated and synthetic; real Synthea patient bundles were not obtainable
in this build environment (no internet access to the data source at build
time), so this is disclosed as an explicit scope decision, the same choice
Phase 4's base generator already made and stated for its own data.

## Results

| Model | AUROC | Note |
|---|---|---|
| Structured XGBoost | 0.742 | baseline |
| + Retrieval features | **0.766** | recovers latent per-provider denial propensity |
| Oracle ceiling | 0.869 | max recoverable given the injected label rule |
| Leaky index (invalid, reported as contrast) | 0.973 | what you'd wrongly report if the index leaked test claims |

Evidence-completeness extension, 150 cases:

| Method | Accuracy |
|---|---|
| TF-IDF (lexical) | 80.0% |
| Semantic (MiniLM) | 82.7% |

The two retrievers make complementary errors: TF-IDF fails on paraphrased,
indirect evidence (radiculopathy described narratively rather than in the
requirement's own wording); the semantic retriever recovers those cases but
under-performs on abstractly-phrased requirements matched against very
concrete clinical text. The semantic retriever's similarity score also
separates cleanly by ground truth (mean 0.525 for evidence-present cases vs.
0.258 for evidence-absent), shown via histogram and confusion matrix
figures rather than asserted.

## Discussion (Phase 4 contribution)

**Label trustworthiness.** Rather than assume the injected labels are
realistic, both the claim-level and requirement-level labels are stress-tested:
a noise sweep quantifies AUROC decay as labels are corrupted, a
recover-the-rule check confirms the model learns the actual injected
mechanism, and — for the evidence extension — a hand-audited 12% sample of
cases (18/150) was checked directly against the generator's own injection
logic, with 100% agreement.

**What retrieval adds, and where it still fails.** Retrieval recovers a
signal (per-provider propensity, requirement-level omission) that a flat
classifier structurally cannot see from a single claim or a single
document read in isolation. The honest limit is equally reported: the
oracle ceiling caps how much any feature-based model can recover given a
partly-stochastic label, and the evidence retriever's failure mode (abstract
phrasing vs. concrete clinical text) is shown, not hidden, since it's a
real property of the method worth reporting on its own.

## Future work (Phase 4)

- Extend the evidence-completeness test beyond 2 procedure families and
  150 cases if time allows.
- A constrained LLM paraphraser for evidence-chunk lexical variety, with
  ground truth still assigned by code (not inferred by the LLM) — keeps
  label validity while adding realism.
- Re-attempt real MIMIC-IV-Note / Synthea credentialing for a real-data
  validation pass over the synthetic result.

---

## AI Prompts Used — Sruthilaya's session (Phase 4 + extension)

*Per the rubric, prompts are logged for the intellectual decisions they drove.*

0. **Recognizing the opportunity to go beyond the assigned Phase 4 scope.**
   Phase 4's deliverable, as scoped, already answered the assignment. The
   decision to build a second, independent extension was not asked for by
   the course or the team plan — it came from reading the instructor's two
   specific feedback points (proxy-label trust, population harmonization)
   and recognizing that Phase 4's own answers to them, while valid at the
   claim level, could be made more rigorous by re-testing the same two
   questions one level deeper, at the level of an individual payer
   requirement. Realizing that was possible required first researching
   real CMS Local Coverage Determination policy documentation (LCD
   L34220/L37281 for lumbar MRI, L33942 for PT/rehab) to find requirements
   concrete enough to build a controlled, gradeable test around — grounding
   the extension in real payer policy rather than inventing an arbitrary
   toy problem. Every subsequent prompt in this log (scoping, the
   data-generation call, what not to build, validation discipline, the RAG
   design reasoning) followed from that initial, self-directed research
   step rather than from an instruction to extend the project.
1. **Scoping the extension idea.** Asked whether extending Phase 4 with a
   second, deeper RAG (claim-level → requirement-level evidence
   completeness) was a legitimate way to address two specific instructor
   feedback points (proxy-label validation, population harmonization)
   without it reading as scope creep — the answer shaped the "same
   retrieval rigor, one level deeper, not a new deviation" framing used
   throughout.
2. **Data-generation design decision.** Asked directly whether to keep the
   existing template-based, ground-truth-injected data generator or switch
   to free-form LLM-generated clinical notes for more surface realism.
   Decided, on academic grounds (not creative preference), to keep the
   controlled generator: it's the standard "programmatic/weak supervision"
   approach in ML literature, preserves an auditable ground truth, and
   avoids re-introducing the exact proxy-label trust problem the extension
   exists to answer. LLM use was scoped down to an optional future-work
   paraphrasing layer, with the ground truth still code-assigned.
3. **Deciding what NOT to build (harmonization scope).** Investigated
   whether to force the evidence extension's 150 text-based cases through
   the existing claim-level PSI/KS harmonization check. Concluded the two
   are different units of analysis (billing fields vs. text evidence) and
   that building fake structured fields just to run the tool would add
   false complexity rather than rigor — feedback #2 is already answered at
   the claim level, and the report states that explicitly instead of
   forcing an artificial comparison.
4. **Validating before building on top.** Before adding downstream
   artifacts (figures, demo tab, report), ran an explicit gold-check: hand
   verified a 12% sample of the 150 generated cases against the injection
   logic, confirming 100% label agreement — the same "stress-test your
   proxy labels early" discipline the instructor asked for, applied one
   level deeper than where Phase 4 already applies it.
5. **Understanding why RAG separates retrieval from generation.** Asked
   directly whether the demo was "showing the RAG part cleanly" — the
   underlying question being what actually makes a system RAG rather than
   a classifier with a caption. The answer: RAG's guarantee is that its
   explanation is *grounded* in a specific, independently-checkable
   retrieved item, and that guarantee only holds if generation is
   restricted to describing what was retrieved rather than reasoning
   freely — otherwise the generation step can hallucinate evidence that
   was never actually retrieved. This is why the report frames retrieval
   and generation as strictly separated stages (see Methods), not just two
   steps in a pipeline diagram.
6. **Understanding retrieval latency and caching as a systems property.**
   Asked why the retrieval demo felt instant and whether that meant
   results were precomputed rather than genuinely retrieved. The answer
   clarified a real distinction in RAG systems: retrieval over a small,
   repeated vocabulary is dominated by embedding-cache hits, while a fresh
   piece of text still requires a real encoder pass — so "fast" does not
   mean "faked," but a demo that doesn't distinguish the two invites that
   exact doubt. This shaped the decision to treat cache-hit-vs-computed as
   a reportable property of the retrieval stage, not a demo cosmetic.
7. **Understanding why the lexical/semantic comparison is the finding.**
   Working through *why* TF-IDF and the semantic retriever fail on
   different cases (IMG-4/5 vs. IMG-2/3) clarified that this isn't just
   "one method beats the other" — it's a lexical-similarity-vs-semantic-
   similarity tradeoff that generalizes beyond this dataset (exact wording
   match vs. learned meaning match), which is why both retrievers are
   reported side by side as the headline result rather than only the
   higher-accuracy one.
