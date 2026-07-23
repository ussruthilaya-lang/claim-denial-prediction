"""
Controlled synthetic claim generator for Phase 4.

WHY A CONTROLLED GENERATOR (and not the raw Synthea jar):
The proposal specifies Synthea + fault-injected denial labels. Synthea itself
produces *no* denial labels — every plan that uses it still has to inject them.
So the modeling story is identical whether the structured fields come from
Synthea or from this generator. What a controlled generator buys us, and Synthea
does not, is a KNOWN ground-truth denial mechanism. Because we author the rule
that turns claim features into a denial (see `labeling.py`), we can do three
things you can never do with real or black-box labels:

  1. Verify the model *recovers* the injected rule (a signal-recovery sanity
     check the prototype's AUROC=0.53 badly failed).
  2. Quantify exactly how label noise degrades every downstream metric
     (the "stress-test your proxy labels" feedback, turned into an experiment).
  3. Keep the whole pipeline reproducible offline on any machine, no Java run,
     no PhysioNet credentialing, no network — which matters on a 2-day clock.

The generated fields map 1:1 onto `shared.schemas.claim.ClaimRecord`, so Phase 4
reads/writes the same typed schema as Phases 1-3 and the cross-phase ablation
stays valid.

The joint distribution is deliberately NOT independent columns: billed amount
depends on the procedure, prior-auth requirement depends on the procedure,
network status depends on the payer, and the filing delay depends on the payer.
Independent columns would make denial trivially predictable and the ablation
meaningless; correlated columns are what make retrieval features earn their keep.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from shared.schemas.claim import ClaimRecord, InsuranceType

# --------------------------------------------------------------------------- #
# Clinical vocabulary. Small on purpose: a realistic biller sees the same few
# hundred code combinations over and over, and a compact vocabulary is what lets
# the FAISS retrieval layer find genuinely "similar past claims."
# --------------------------------------------------------------------------- #

# ICD-10 primary diagnosis codes, grouped into clinical categories. The category
# is what drives medical-necessity (ICD/CPT compatibility), not the exact code.
ICD10 = {
    "chronic": ["E11.9", "I10", "E78.5", "N18.3"],          # diabetes, HTN, lipids, CKD
    "respiratory": ["J45.909", "J44.9", "J20.9"],            # asthma, COPD, bronchitis
    "musculoskeletal": ["M54.5", "M25.561", "M17.11"],       # low-back, knee pain, OA
    "mental_health": ["F41.1", "F32.9", "F43.23"],           # anxiety, depression, adj.
    "gastro": ["K21.9", "K57.30", "K80.20"],                 # GERD, diverticulosis, gallstones
}
ICD_CATEGORIES = list(ICD10)

# CPT procedure codes with the metadata the denial rule needs:
#   base_cost         : typical billed amount (lognormal center)
#   prior_auth        : does this procedure normally require prior authorization?
#   covered_for       : ICD categories for which this procedure is medically
#                       necessary; a claim whose diagnosis is outside this set
#                       is a medical-necessity mismatch and denial-prone.
@dataclass(frozen=True)
class Procedure:
    cpt: str
    label: str
    base_cost: float
    prior_auth: bool
    covered_for: tuple[str, ...]


PROCEDURES = [
    Procedure("99213", "Office visit, low", 120, False, tuple(ICD_CATEGORIES)),
    Procedure("99214", "Office visit, moderate", 200, False, tuple(ICD_CATEGORIES)),
    Procedure("93000", "ECG", 180, False, ("chronic",)),
    Procedure("80053", "Metabolic panel", 90, False, ("chronic", "gastro")),
    Procedure("70450", "CT head w/o contrast", 1200, True, ("mental_health", "respiratory")),
    Procedure("45378", "Colonoscopy", 2100, True, ("gastro",)),
    Procedure("20610", "Major joint injection", 350, True, ("musculoskeletal",)),
    Procedure("97110", "Physical therapy", 110, False, ("musculoskeletal",)),
    Procedure("90837", "Psychotherapy, 60min", 190, True, ("mental_health",)),
    Procedure("71046", "Chest X-ray", 260, False, ("respiratory",)),
]
PROCEDURE_BY_CPT = {p.cpt: p for p in PROCEDURES}

INSURANCE_TYPES = [
    InsuranceType.MEDICARE.value,
    InsuranceType.MEDICAID.value,
    InsuranceType.PRIVATE.value,
    InsuranceType.SELF_PAY.value,
    InsuranceType.OTHER.value,
]
# Prevalence of each payer in the book of business (roughly US-representative).
INSURANCE_WEIGHTS = np.array([0.30, 0.20, 0.38, 0.07, 0.05])

# Payer-specific timely-filing windows (days) after which a claim is denied for
# late filing. Medicaid famously runs tight windows; self-pay has none.
FILING_WINDOW_DAYS = {
    "medicare": 365,
    "medicaid": 95,
    "private": 180,
    "self_pay": 10_000,   # effectively no payer filing rule
    "other": 180,
}

N_PROVIDERS = 60

# --------------------------------------------------------------------------- #
# Clinical-note synthesis (Phase 3). The note is generated from the LATENT
# `_necessity_documented` flag and NEVER from the denial label. Its only
# label-relevant content is whether medical-necessity justification is
# documented — a signal that lives in the narrative and is deliberately kept out
# of the structured billing fields, so a text model (Phase 3) can recover
# something the flat structured model cannot see. This is the text-side
# counterpart to the latent per-provider propensity that only retrieval
# (Phase 4) can reach. Wording is varied and lossy (templates + synonyms) so the
# signal is *expressed*, not stamped as a single give-away token — which is what
# keeps the recoverable AUROC realistic rather than extreme.
# --------------------------------------------------------------------------- #
_DX_PHRASES = {
    "chronic": ["type 2 diabetes mellitus", "essential hypertension",
                "hyperlipidemia", "stage 3 chronic kidney disease"],
    "respiratory": ["an asthma exacerbation", "chronic obstructive pulmonary disease",
                    "acute bronchitis"],
    "musculoskeletal": ["chronic low back pain", "knee osteoarthritis",
                        "persistent joint pain"],
    "mental_health": ["generalized anxiety disorder", "major depressive disorder",
                      "an adjustment disorder"],
    "gastro": ["gastroesophageal reflux disease", "diverticulosis", "cholelithiasis"],
}
_SEX = ["male", "female"]
_FILLER = [
    "Vital signs were stable and the examination was unremarkable aside from the presenting complaint.",
    "Past medical history was reviewed and home medications were reconciled.",
    "The patient tolerated the encounter well and appropriate follow-up was arranged.",
    "Relevant laboratory studies were reviewed and were within expected limits.",
]
# Documented-necessity language (present only when _necessity_documented True).
_JUSTIFICATION = [
    "Conservative management including NSAIDs and physical therapy was trialed for six weeks without adequate relief.",
    "Symptoms have progressed despite first-line therapy, and prior imaging supports the need for this intervention.",
    "The patient meets published guideline criteria; earlier conservative measures failed to control symptoms.",
    "Documentation establishes medical necessity: prior treatments were exhausted and the condition continues to worsen.",
]
# Text used when necessity is NOT documented — a neutral procedure line or an
# explicit documentation gap. Never mentions coverage, payer, or the outcome.
_NO_JUSTIFICATION = [
    "The procedure was performed during today's encounter.",
    "The service was carried out as scheduled.",
    "No prior conservative therapy is documented in the available record.",
    "Indication was noted per the ordering provider; further justification was not recorded.",
]


def _compose_note(aux, icd_cat: str, proc_label: str, documented: bool) -> str:
    """Build one synthetic clinical note. `aux` is a dedicated RNG stream so note
    text does not perturb the structured-claim random draws (keeps the structured
    fields identical to the note-free baseline for clean regression checks)."""
    age = int(aux.integers(19, 89))
    sex = _SEX[aux.integers(2)]
    dx_list = _DX_PHRASES[icd_cat]
    dx = dx_list[aux.integers(len(dx_list))]
    verb = ["performed", "ordered", "completed"][aux.integers(3)]
    opening = f"{age}-year-old {sex} evaluated for {dx}."
    proc_line = f"{proc_label} was {verb}."
    if documented:
        necessity = _JUSTIFICATION[aux.integers(len(_JUSTIFICATION))]
    else:
        necessity = _NO_JUSTIFICATION[aux.integers(len(_NO_JUSTIFICATION))]
    fi = aux.choice(len(_FILLER), size=2, replace=False)
    parts = [opening, _FILLER[fi[0]], proc_line, necessity, _FILLER[fi[1]]]
    return " ".join(parts)


@dataclass
class GeneratedClaims:
    """Bundle: the observable claim frame plus the latent driver columns the
    labeling rule consumes. Kept separate so `features.py` can be handed only
    the observable frame and can't accidentally train on a latent driver that a
    real biller would not have seen."""

    frame: pd.DataFrame           # observable + latent columns (latent prefixed `_`)
    observable_columns: list[str]
    latent_columns: list[str]


def generate_claims(n: int = 40_000, seed: int = 42,
                    start: date = date(2023, 1, 1),
                    end: date = date(2024, 12, 31)) -> GeneratedClaims:
    """Generate `n` synthetic claims with correlated, realistic structure.

    Latent columns (prefixed with `_`) encode the denial drivers and are
    consumed only by `labeling.py`; the observable columns are what a billing
    reviewer — and therefore the model and the retriever — actually see.
    """
    rng = np.random.default_rng(seed)

    # Providers vary in "clean claim" discipline: a provider's baseline documents
    # -and-obtains-prior-auth rate. This is what makes provider_id predictive and
    # gives the retriever a real cluster structure to exploit.
    provider_ids = [f"prov-{i:03d}" for i in range(N_PROVIDERS)]
    provider_quality = dict(zip(provider_ids, rng.beta(6, 2, size=N_PROVIDERS)))
    # Fraction of each provider's volume that is in-network for a given payer.
    provider_network = dict(zip(provider_ids, rng.beta(8, 2, size=N_PROVIDERS)))
    # Systematic per-provider denial propensity (log-odds) beyond the observable
    # drivers: some practices simply have worse coding/documentation habits. This
    # is a HIGH-CARDINALITY, LATENT signal — the flat classifier never sees
    # provider_id, so it cannot use it, but provider-clustered retrieval can.
    provider_propensity = dict(zip(provider_ids, rng.normal(0.0, 0.9, N_PROVIDERS)))

    span_days = (end - start).days
    rows = []
    # Dedicated RNG for note text + documentation flag, so the structured claim
    # draws below stay byte-identical to the note-free baseline.
    aux = np.random.default_rng(seed + 7)
    for i in range(n):
        payer = rng.choice(INSURANCE_TYPES, p=INSURANCE_WEIGHTS)
        icd_cat = rng.choice(ICD_CATEGORIES)
        icd10 = rng.choice(ICD10[icd_cat])
        provider = provider_ids[rng.integers(N_PROVIDERS)]

        # Procedure choice is mostly clinically appropriate for the diagnosis
        # (a biller rarely orders a colonoscopy for asthma). ~20% of claims are
        # deliberately mismatched, so medical-necessity denial stays a real but
        # minority driver rather than firing on the majority of claims.
        compatible = [p for p in PROCEDURES if icd_cat in p.covered_for]
        if compatible and rng.random() < 0.80:
            proc = compatible[rng.integers(len(compatible))]
        else:
            proc = PROCEDURES[rng.integers(len(PROCEDURES))]

        # Billed amount: lognormal around the procedure's base cost. A minority of
        # claims are "upcoded" (amount well above the procedure norm) — a real
        # denial driver (billing anomaly / documentation-not-supporting-level).
        upcoded = rng.random() < 0.08
        mult = rng.lognormal(mean=0.0, sigma=0.25) * (2.2 if upcoded else 1.0)
        billed = round(float(proc.base_cost * mult), 2)

        # Dates: service date uniform in window; filing delay is payer-shaped
        # (Medicaid submits faster on average; a tail files late and gets denied).
        service = start + timedelta(days=int(rng.integers(0, span_days + 1)))
        base_delay = {"medicaid": 20, "medicare": 40, "private": 30,
                      "self_pay": 25, "other": 30}[payer]
        filing_days = int(max(0, rng.gamma(shape=2.0, scale=base_delay / 2.0)))
        # ~6% of claims get "lost" and refiled very late (staff turnover, denied-
        # and-resubmitted loops), which is what actually trips timely-filing.
        if rng.random() < 0.06:
            filing_days += int(rng.gamma(shape=3.0, scale=120.0))
        submission = service + timedelta(days=filing_days)

        # Prior auth: required by the procedure; obtained depends on provider
        # discipline. Missing-when-required is a top denial reason nationally.
        pa_required = proc.prior_auth
        pa_obtained = bool(rng.random() < provider_quality[provider]) if pa_required else True

        # Network status: depends on provider's network breadth for this payer.
        in_network = bool(rng.random() < provider_network[provider])

        # Medical-necessity match: is the diagnosis category one this procedure
        # is covered for?
        necessity_ok = icd_cat in proc.covered_for

        # Documentation-of-necessity: latent, orthogonal to the observable
        # billing fields, expressed only in the clinical note. ~60% of encounters
        # document justification; when they don't, denial risk rises (see
        # `undocumented_necessity` in labeling.py). This is the note-only signal
        # Phase 3 exists to recover.
        necessity_documented = bool(aux.random() < 0.60)
        clinical_note = _compose_note(aux, icd_cat, proc.label, necessity_documented)

        rows.append({
            "claim_id": f"CLM-{i:07d}",
            "provider_id": provider,
            "patient_id": f"PT-{rng.integers(0, n // 3):07d}",
            "icd10_code": icd10,
            "cpt_code": proc.cpt,
            "insurance_type": payer,
            "billed_amount": billed,
            "service_date": service,
            "submission_date": submission,
            "clinical_note": clinical_note,
            # ---- latent denial drivers (consumed by labeling.py only) ----
            "_icd_category": icd_cat,
            "_procedure_base_cost": proc.base_cost,
            "_amount_ratio": billed / proc.base_cost,
            "_filing_days": filing_days,
            "_filing_window": FILING_WINDOW_DAYS[payer],
            "_prior_auth_required": pa_required,
            "_prior_auth_obtained": pa_obtained,
            "_in_network": in_network,
            "_necessity_ok": necessity_ok,
            "_necessity_documented": necessity_documented,
            "_provider_quality": provider_quality[provider],
            "_provider_propensity": provider_propensity[provider],
        })

    frame = pd.DataFrame(rows)
    latent = [c for c in frame.columns if c.startswith("_")]
    observable = [c for c in frame.columns if not c.startswith("_")]
    return GeneratedClaims(frame=frame, observable_columns=observable,
                           latent_columns=latent)


def row_to_claim(row: pd.Series) -> ClaimRecord:
    """Convert one observable row to the shared typed schema. Used by the
    retriever's metadata store and the demo app so Phase 4 speaks the same
    ClaimRecord contract as every other phase."""
    return ClaimRecord(
        claim_id=row["claim_id"],
        provider_id=row["provider_id"],
        patient_id=row.get("patient_id"),
        icd10_code=row["icd10_code"],
        cpt_code=row["cpt_code"],
        insurance_type=row["insurance_type"],
        billed_amount=float(row["billed_amount"]),
        service_date=row["service_date"],
        submission_date=row["submission_date"],
        reason_code=row.get("reason_code") if pd.notna(row.get("reason_code")) else None,
        denied=bool(row["denied"]) if "denied" in row and pd.notna(row["denied"]) else None,
    )


def generate_dataset(n: int = 40_000, seed: int = 42,
                     target_prevalence: float = 0.19,
                     label_noise: float = 0.0) -> pd.DataFrame:
    """Unified entry point for ALL phases: generate correlated claims (with a
    synthetic `clinical_note` per claim) AND inject denial labels in one call, so
    every phase consumes the exact same rows and the cross-phase ablation is
    apples-to-apples by construction.

    Returns the labeled frame with observable columns (incl. `clinical_note`),
    the `denied` / `true_denial_prob` / `reason_code` labels, and the latent
    `_`-prefixed driver columns (consumed only by labeling / auditing, never fed
    to a model). Import is function-local to avoid a data_gen<->labeling cycle.
    """
    from phase4_rag_agentic.src.labeling import label_claims
    gen = generate_claims(n=n, seed=seed)
    return label_claims(gen.frame, target_prevalence=target_prevalence,
                        label_noise=label_noise, seed=seed).frame


if __name__ == "__main__":
    gen = generate_claims(n=2000, seed=0)
    print(gen.frame[gen.observable_columns].head())
    print("\nlatent drivers:", gen.latent_columns)
    print("n claims:", len(gen.frame))
    doc_rate = (~gen.frame["_necessity_documented"]).mean()
    print(f"\nundocumented-necessity rate: {doc_rate:.3f}")
    print("\nsample clinical_note:\n", gen.frame["clinical_note"].iloc[0])
