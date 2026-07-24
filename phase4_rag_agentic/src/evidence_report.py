"""
Decision-support layer for the evidence-completeness extension — the
requirement-level counterpart to `llm_demo.py`'s claim-level decision support.

WHY THIS IS THE PRODUCT, NOT A GARNISH:
"This claim is 82% likely to be denied" tells a reviewer nothing to act on.
"The lumbar-MRI conservative-treatment requirement is not supported — the PT
progress note documenting 6 weeks of therapy is in the chart but was not
submitted with this claim" tells them exactly what to attach before
resubmission. That is the deliverable this extension exists to produce.

DESIGN — status and citation from the RETRIEVER, prose from the LLM:
The evidence status (`complete` / `omitted` / `unsupported`) and the cited
chunk always come from the semantic retriever (deterministic, evaluated at
82.7% accuracy). The LLM's job is narrow: turn that structured result into a
one-paragraph reviewer-facing explanation. If no API key is configured, a
deterministic `mock_llm` composes the same explanation directly from the
retriever's output — so the demo and grading run with zero external
dependency, and a real key only upgrades the prose, never the underlying
status.
"""
from __future__ import annotations

from dataclasses import dataclass

from phase4_rag_agentic.src.evidence_retriever_semantic import classify_case
from shared.config.settings import settings

SYSTEM_PROMPT = (
    "You are a medical-billing documentation reviewer. Given a payer "
    "requirement, its evidence status, and (if found) the cited chart note, "
    "write one concise paragraph explaining the status to a billing reviewer "
    "and what to do next. Do not invent clinical facts beyond what is cited."
)

ACTION_BY_STATUS = {
    "complete": "No action needed — the requirement is documented and was submitted with the claim.",
    "omitted": "Attach the cited note to the submission before resubmitting; the evidence exists in the record but was left out of the packet.",
    "unsupported": "Obtain and document this requirement in the chart before submitting the claim; no supporting evidence was found in the record.",
}


@dataclass
class DeficiencyReport:
    case_id: int
    req_id: str
    requirement_text: str
    status: str
    cited_chunk: str | None
    confidence: float
    suggested_action: str
    rationale: str


def mock_llm(requirement_text: str, status: str, cited_chunk: str | None,
             confidence: float) -> str:
    """Deterministic rationale composed from the retriever's own output. No network."""
    if status == "complete":
        return (f"Requirement met: \"{requirement_text}\" Evidence was found in the "
                f"submitted packet (match confidence {confidence:.2f}): "
                f"\"{cited_chunk}\" No further documentation is needed for this item.")
    if status == "omitted":
        return (f"Requirement not submitted: \"{requirement_text}\" Supporting "
                f"evidence exists in the patient's full record (match confidence "
                f"{confidence:.2f}) but was not included in what was submitted with "
                f"this claim: \"{cited_chunk}\" Adding this note to the packet "
                f"should resolve the deficiency without needing a new encounter.")
    return (f"Requirement unsupported: \"{requirement_text}\" No matching evidence "
            f"was found anywhere in the patient's record (best match confidence "
            f"{confidence:.2f}, below the threshold for a credible match). "
            f"Documentation supporting this requirement will need to be obtained "
            f"before the claim can be submitted with confidence.")


def _real_llm(system: str, user: str) -> str | None:
    """Optional: use a configured LLM to write the rationale. Returns None if no
    key/library is available, so callers fall back to `mock_llm`."""
    if settings.anthropic_api_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            msg = client.messages.create(
                model="claude-sonnet-5", max_tokens=250,
                system=system, messages=[{"role": "user", "content": user}])
            return msg.content[0].text
        except Exception:
            return None
    if settings.openai_api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=settings.openai_api_key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini", max_tokens=250,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}])
            return resp.choices[0].message.content
        except Exception:
            return None
    return None


def _user_prompt(requirement_text: str, status: str, cited_chunk: str | None) -> str:
    return (f"Requirement: {requirement_text}\nEvidence status: {status}\n"
            f"Cited chart note: {cited_chunk or '(none found)'}\n"
            f"Write the reviewer-facing explanation.")


def explain_case(case: dict, use_real_llm: bool = False) -> DeficiencyReport:
    """Full decision-support path for one evidence case: classify -> cite -> explain.

    `case` is one entry from `evidence_data_gen.generate_cases()`."""
    result = classify_case(case)
    status = result["predicted_status"]
    cited_chunk = result["cited_chunk"]
    confidence = result["sub_score"] if status == "complete" else result["rec_score"]

    user_prompt = _user_prompt(case["requirement_text"], status, cited_chunk)
    rationale = (_real_llm(SYSTEM_PROMPT, user_prompt) if use_real_llm else None) \
        or mock_llm(case["requirement_text"], status, cited_chunk, confidence)

    return DeficiencyReport(
        case_id=case["case_id"], req_id=case["req_id"],
        requirement_text=case["requirement_text"], status=status,
        cited_chunk=cited_chunk, confidence=confidence,
        suggested_action=ACTION_BY_STATUS[status], rationale=rationale,
    )
