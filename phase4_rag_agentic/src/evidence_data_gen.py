"""
Generates synthetic-but-schema-realistic longitudinal patient chunks and builds
labeled (requirement, submission, full-record) cases with exact injected ground
truth. Structurally mirrors Synthea's encounter/observation model; real Synthea
bundles were not downloadable in this environment (see data provenance note).
"""
import random
from src.evidence_policies import ALL_POLICIES

# positive evidence + distractor chunk templates per requirement id
EVIDENCE_TEMPLATES = {
    "IMG-1": ("PT progress note: patient completed {n} weeks of physical therapy for low back pain, "
              "including a supervised exercise program.", 6),
    "IMG-2": ("Clinic note: prior conservative treatment history reviewed and documented; "
              "symptoms persist despite NSAID trial.", None),
    "IMG-3": ("Intake note: low back pain present for {n} weeks, described as dull and worsening with sitting.", 8),
    "IMG-4": ("Neuro exam note: radiculopathy with numbness radiating down the left leg, positive straight-leg raise.", None),
    "IMG-5": ("Functional assessment: patient reports difficulty standing >10 minutes; gait exam shows antalgic pattern.", None),
    "PT-1": ("Progress note dated visit #{n}: therapist reviewed goals and functional status per required interval.", 10),
    "PT-2": ("Objective measures: shoulder flexion improved from 90 to 130 degrees; grip strength increased 15%.", None),
    "PT-3": ("Reassessment note: no functional improvement after {n} sessions; treatment plan modified to add manual therapy.", 6),
    "PT-4": ("Justification note: continued therapy beyond initial visit allotment supported by persistent functional deficit.", None),
    "PT-5": ("Plan of care signed and certified by Dr. Okafor, physician, on file.", None),
}

DISTRACTOR_CHUNKS = [
    "Vitals: BP 128/82, HR 76, temp 98.4F, recorded at intake.",
    "Billing note: copay collected at time of visit.",
    "Medication list reviewed and reconciled, no changes.",
    "Patient reports no known drug allergies.",
    "Front desk note: appointment rescheduled once due to weather.",
]

def _evidence_text(req_id):
    tpl, n = EVIDENCE_TEMPLATES[req_id]
    return tpl.format(n=n) if n else tpl

def generate_cases(patients_per_requirement=5, seed=42):
    rng = random.Random(seed)
    cases = []
    case_id = 0
    for policy in ALL_POLICIES:
        for req in policy["requirements"]:
            for p in range(patients_per_requirement):
                for variant in ["complete", "omitted", "unsupported"]:
                    patient_id = f"{policy['family']}_{req['req_id']}_p{p}"
                    evidence_chunk = _evidence_text(req["req_id"])
                    distractors = rng.sample(DISTRACTOR_CHUNKS, k=3)

                    if variant == "complete":
                        full_record = distractors + [evidence_chunk]
                        submitted = [evidence_chunk]
                        gold_evidence_present = True
                    elif variant == "omitted":
                        full_record = distractors + [evidence_chunk]
                        submitted = distractors[:1]  # evidence exists but not submitted
                        gold_evidence_present = True
                    else:  # unsupported
                        full_record = distractors
                        submitted = distractors[:1]
                        gold_evidence_present = False

                    cases.append({
                        "case_id": case_id,
                        "policy_id": policy["policy_id"],
                        "req_id": req["req_id"],
                        "requirement_text": req["text"],
                        "patient_id": patient_id,
                        "submitted_chunks": submitted,
                        "full_record_chunks": full_record,
                        "gold_variant": variant,  # complete | omitted | unsupported
                        "gold_evidence_text": evidence_chunk if gold_evidence_present else None,
                    })
                    case_id += 1
    return cases