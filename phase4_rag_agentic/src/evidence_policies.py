"""
Real CMS LCD-derived requirement sentences for the 2 procedure families.
Text is paraphrased from public CMS LCD / CMS documentation-standard sources
(see `source` field), not copied verbatim, per each requirement.
"""

IMAGING_POLICY = {
    "policy_id": "LCD_L34220_LUMBAR_MRI",
    "family": "advanced_imaging",
    "source": "CMS LCD L34220 / L37281 - Lumbar MRI",
    "requirements": [
        {"req_id": "IMG-1", "text": "At least 6 weeks of conservative treatment, including physical therapy, must be documented before advanced imaging is considered."},
        {"req_id": "IMG-2", "text": "Clinical findings and prior treatment history supporting the need for the MRI must be documented in the clinical record."},
        {"req_id": "IMG-3", "text": "Duration and character of symptoms must be documented, not just a general referral reason."},
        {"req_id": "IMG-4", "text": "If neurological symptoms (e.g., radiculopathy) are present, they must be explicitly documented as an indication."},
        {"req_id": "IMG-5", "text": "Imaging findings must be correlated with documented functional limitations or physical exam findings."},
    ],
}

PT_POLICY = {
    "policy_id": "LCD_L33942_PT_CONTINUED",
    "family": "pt_rehab",
    "source": "CMS LCD L33942 / Medicare Part B therapy documentation standards",
    "requirements": [
        {"req_id": "PT-1", "text": "A progress note is required at least every 10 treatment visits or 30 calendar days, whichever comes first."},
        {"req_id": "PT-2", "text": "Progress notes must include objective measures of functional improvement (e.g., strength, range of motion)."},
        {"req_id": "PT-3", "text": "If no objective or subjective improvement is noted after a defined number of treatments, the record must document a change in treatment plan or justification for continuing."},
        {"req_id": "PT-4", "text": "Continued treatment beyond the initial visit allotment requires documentation supporting medical necessity for the extension."},
        {"req_id": "PT-5", "text": "The plan of care must be certified/signed by the physician or qualified practitioner."},
    ],
}

ALL_POLICIES = [IMAGING_POLICY, PT_POLICY]