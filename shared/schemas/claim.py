"""
Canonical claim record schema.

WHY THIS FILE EXISTS:
Phase 1 uses Kaggle's synthetic claims (billed_amount, provider_id, procedure_code...).
Phase 3 needs to fuse those fields with ClinicalBERT embeddings from MIMIC-IV-Note.
Phase 4 needs the same fields to retrieve "similar past claims" via FAISS.
If each phase defines its own dataframe columns ad hoc, Phase 4's retrieval index
will silently drift out of sync with what Phase 1/2 trained on — and nobody notices
until the ablation study numbers don't line up. One typed schema, enforced everywhere,
prevents that class of bug entirely.

Using pydantic (not just a dict/dataframe convention) means malformed records fail
loudly at ingestion time instead of silently corrupting a downstream model.
"""
from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class InsuranceType(str, Enum):
    MEDICARE = "medicare"
    MEDICAID = "medicaid"
    PRIVATE = "private"
    SELF_PAY = "self_pay"
    OTHER = "other"


class ClaimRecord(BaseModel):
    """
    One medical claim. Structured fields map to Phase 1/2's Kaggle dataset;
    `clinical_note_text` is populated only for records that also exist in
    MIMIC-IV-Note (Phase 3), and stays None otherwise.
    """

    model_config = ConfigDict(use_enum_values=True)

    claim_id: str
    provider_id: str
    patient_id: Optional[str] = None  # None for fully de-identified sources

    icd10_code: str = Field(..., description="Primary diagnosis code")
    cpt_code: str = Field(..., description="Primary procedure code")
    insurance_type: InsuranceType
    billed_amount: float = Field(..., ge=0)

    service_date: date
    submission_date: date

    reason_code: Optional[str] = None
    clinical_note_text: Optional[str] = Field(
        default=None,
        description="Discharge summary / physician note, Phase 3+ only. "
        "Never populate from raw MIMIC-IV-Note without CITI-credentialed access.",
    )

    denied: Optional[bool] = Field(
        default=None, description="Label. None at inference time."
    )

    @field_validator("submission_date")
    @classmethod
    def submission_after_service(cls, v: date, info):
        service_date = info.data.get("service_date")
        if service_date and v < service_date:
            raise ValueError("submission_date cannot precede service_date")
        return v
