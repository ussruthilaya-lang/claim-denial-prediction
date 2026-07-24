"""
DenialGuard — Phase 4 decision-support demo (the "working application").

Plain-language, task-oriented UI with progressive disclosure: each tab leads
with the answer a non-expert needs, and pushes the ML detail (SHAP,
calibration, PSI, similar-claims table, retrieval scores) behind expanders so
a technical reviewer can still drill in.

Three tabs:
  1. Check a claim        — denial risk + why + suggested fix (retrieval-augmented).
  2. How well it works    — accuracy / cost / savings, with full diagnostics on demand.
  3. Check the documentation — the evidence-completeness extension: is a payer
     requirement supported by the record, and was that evidence submitted?

Run:  streamlit run mlops_platform/demo/app.py
Needs the Phase 4 artifacts:  python scripts/run_phase4.py   (writes them)
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from phase4_rag_agentic.src.data_gen import ICD10, PROCEDURE_BY_CPT, row_to_claim
from phase4_rag_agentic.src.evidence_data_gen import generate_cases
from phase4_rag_agentic.src.evidence_policies import ALL_POLICIES
from phase4_rag_agentic.src.evidence_report import explain_case
from phase4_rag_agentic.src.evidence_retriever import build_global_index
from phase4_rag_agentic.src.evidence_retriever import classify_case as tfidf_classify
from phase4_rag_agentic.src.evidence_retriever_semantic import \
    cache_size as semantic_cache_size
from phase4_rag_agentic.src.evidence_retriever_semantic import \
    classify_case as semantic_classify
from phase4_rag_agentic.src.features import RetrievalFeaturizer
from phase4_rag_agentic.src.llm_demo import explain_claim
from phase4_rag_agentic.src.retriever import ClaimRetriever

ART = ROOT / "phase4_rag_agentic" / "artifacts"
FIG = ART / "figures"

# Claim-level risk bands (Score a claim).
BAND_COLOR = {"HIGH": "#d03b3b", "ELEVATED": "#fab219", "LOW": "#0ca30c"}
BAND_LABEL = {"HIGH": "High risk", "ELEVATED": "Elevated risk", "LOW": "Low risk"}
VERDICT_TEXT = {
    "HIGH": "Likely to be denied — review and fix it before submitting.",
    "ELEVATED": "Some denial risk — a quick review is worth it.",
    "LOW": "Looks clean — likely to be paid.",
}
# Evidence-completeness statuses (Check the documentation).
STATUS_COLOR = {"complete": "#0ca30c", "omitted": "#fab219", "unsupported": "#d03b3b"}
STATUS_LABEL = {"complete": "Documentation complete",
                "omitted": "Evidence omitted",
                "unsupported": "Not supported"}
FAMILY_LABEL = {"advanced_imaging": "Advanced imaging", "pt_rehab": "PT / rehab"}

st.set_page_config(page_title="DenialGuard", page_icon="🛡️", layout="wide")


def _inject_css():
    # Minimal declutter: hide Streamlit's Deploy button / toolbar / footer chrome.
    st.markdown(
        "<style>"
        "[data-testid='stToolbar']{visibility:hidden;height:0;}"
        "[data-testid='stDeployButton']{display:none;}"
        "[data-testid='stHeader']{background:transparent;}"
        "footer{visibility:hidden;}"
        "</style>",
        unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    with open(ART / "bundle.pkl", "rb") as f:
        bundle = pickle.load(f)
    retriever = ClaimRetriever.load(ART / "claims_index.faiss")
    art = SimpleNamespace(
        encoder=bundle["encoder"],
        model_augmented=bundle["model_augmented"],
        retriever=retriever,
        featurizer=RetrievalFeaturizer(retriever, k=bundle["k"]),
        cost_operating_point=bundle["cost_operating_point"],
    )
    test = pd.read_parquet(ART / "test_predictions.parquet")
    return art, test


def _icd_options():
    return [c for codes in ICD10.values() for c in codes]


# ───────────────────────────── Tab 1: Check a claim ─────────────────────────
def score_tab(art, test):
    st.caption("Estimate how likely a claim is to be **denied** before you submit it — "
               "with the reason and a suggested fix.")
    left, right = st.columns([1, 1.4], gap="large")

    with left:
        mode = st.radio("Start from", ["A sample claim", "Build my own"],
                        horizontal=True)
        if mode == "A sample claim":
            sample = test.sort_values("prob_augmented", ascending=False).head(200)
            options = sample["claim_id"].tolist()
            cid = st.selectbox("Sample claim (sorted by model risk)", options)
            row = test[test["claim_id"] == cid].iloc[0]
            claim = row_to_claim(row)
        else:
            payer = st.selectbox("Insurance", art.encoder.payers)
            cpt = st.selectbox("Procedure (CPT)",
                               [f"{c} — {PROCEDURE_BY_CPT[c].label}" for c in art.encoder.cpts])
            cpt = cpt.split(" — ")[0]
            icd = st.selectbox("Diagnosis (ICD-10)", _icd_options())
            provider = st.selectbox("Provider", art.encoder.providers)
            billed = st.number_input("Billed amount ($)", 20.0, 20000.0,
                                     float(PROCEDURE_BY_CPT[cpt].base_cost))
            svc = st.date_input("Service date", value=pd.Timestamp("2024-06-01"))
            days = st.slider("Days until submitted", 0, 400, 30)
            sub = pd.Timestamp(svc) + pd.Timedelta(days=days)
            claim = row_to_claim(pd.Series({
                "claim_id": "LIVE-0001", "provider_id": provider,
                "patient_id": None, "icd10_code": icd, "cpt_code": cpt,
                "insurance_type": payer, "billed_amount": billed,
                "service_date": pd.Timestamp(svc), "submission_date": sub,
                "reason_code": None, "denied": None}))

        with st.expander("Claim details", expanded=False):
            filing_days = (claim.submission_date - claim.service_date).days
            fields = pd.DataFrame([
                {"field": "Claim ID", "value": claim.claim_id},
                {"field": "Provider", "value": claim.provider_id},
                {"field": "Diagnosis (ICD-10)", "value": claim.icd10_code},
                {"field": "Procedure (CPT)",
                 "value": f"{claim.cpt_code} — {PROCEDURE_BY_CPT[claim.cpt_code].label}"},
                {"field": "Insurance", "value": claim.insurance_type},
                {"field": "Billed amount", "value": f"${claim.billed_amount:,.2f}"},
                {"field": "Service date", "value": str(claim.service_date)},
                {"field": "Submission date", "value": str(claim.submission_date)},
                {"field": "Days to file", "value": str(filing_days)},
            ])
            st.dataframe(fields, use_container_width=True, hide_index=True,
                        column_config={"field": st.column_config.TextColumn(width="medium"),
                                       "value": st.column_config.TextColumn(width="medium")})

        go = st.button("Assess denial risk", type="primary", use_container_width=True)

    with right:
        if not go:
            st.info("👈 Choose or build a claim, then **Assess denial risk**. "
                    "You'll get a risk score, the reason, and what to fix.")
            return
        ds = explain_claim(art, claim)
        color = BAND_COLOR[ds.risk_band]
        st.markdown(
            f"<div style='padding:18px;border-radius:12px;background:{color}18;"
            f"border:1px solid {color}'>"
            f"<span style='font-size:2.8rem;font-weight:800;color:{color}'>"
            f"{ds.denial_probability:.0%}</span> "
            f"<span style='font-size:1.15rem;color:{color};font-weight:700'>"
            f"denial risk · {BAND_LABEL[ds.risk_band]}</span>"
            f"<div style='margin-top:6px;font-size:1.02rem'>{VERDICT_TEXT[ds.risk_band]}</div>"
            f"</div>",
            unsafe_allow_html=True)
        st.markdown(f"**✅ What to do:** {ds.suggested_action}")
        st.markdown(f"**💡 Why:** {ds.rationale}")

        with st.expander("See the evidence — similar past claims"):
            ev = pd.DataFrame([{
                "claim_id": e.claim.claim_id, "similarity": round(e.similarity, 3),
                "provider": e.claim.provider_id, "payer": e.claim.insurance_type,
                "cpt": e.claim.cpt_code, "billed": e.claim.billed_amount,
                "outcome": "DENIED" if e.claim.denied else "paid",
                "reason": e.claim.reason_code or "",
            } for e in ds.evidence])
            st.dataframe(ev, use_container_width=True, hide_index=True)
            if ds.top_reasons:
                st.caption("Most common denial reasons among these similar claims: "
                           + " · ".join(f"{r} ({n})" for r, n in ds.top_reasons))


# ─────────────────────────── Tab 2: How well it works ───────────────────────
def results_tab(art):
    st.caption("How accurate DenialGuard is, and what it saves — measured on a held-out test set.")
    cop = art.cost_operating_point
    c1, c2, c3 = st.columns(3)
    c1.metric("Review cutoff", f"{cop['threshold']:.0%}",
              help="A claim is flagged for review when its predicted denial risk is above this.")
    c2.metric("Avg. cost per claim", f"${cop['cost_per_claim']:.2f}",
              help="Expected review + denial cost per claim at this cutoff.")
    c3.metric("Savings vs. no model", f"${cop['savings_vs_do_nothing']:,.0f}",
              help="Total saved across the test set vs. reviewing nothing.")

    st.markdown("**Does adding retrieval actually help?**")
    p = FIG / "ablation_auroc.png"
    if p.exists():
        st.image(str(p), use_column_width=True,
                 caption="Structured billing data + retrieval flags more denials than either "
                         "alone (higher AUROC = better; the dashed line is the best achievable).")

    with st.expander("Technical detail — full model diagnostics (for reviewers)"):
        figs = [("shap_bar.png", "What drives the prediction (green = retrieval features)"),
                ("calibration.png", "Calibration — predicted vs. actual denial rate"),
                ("cost_curve.png", "Cost-sensitive operating point"),
                ("noise_sweep.png", "Robustness when labels are noisy"),
                ("harmonization_psi.png", "Population stability, train vs. test (PSI)")]
        cols = st.columns(2)
        for i, (fname, caption) in enumerate(figs):
            fp = FIG / fname
            if fp.exists():
                cols[i % 2].image(str(fp), caption=caption, use_column_width=True)


# ─────────────────────── Tab 3: Check the documentation ─────────────────────
@st.cache_resource
def load_evidence_cases():
    cases = generate_cases()
    build_global_index(cases)  # fits the TF-IDF vectorizer once, globally
    return cases


def evidence_tab():
    st.caption("A separate check: does a claim's documentation actually support the insurer's "
               "requirement — and was that evidence submitted, or left out?")
    with st.expander("About this test bed (30-second scope)"):
        st.write(
            "A separate, narrower test from the denial model in the other tabs: for one "
            "specific payer requirement, is it supported by the patient's record — and if so, "
            "was that evidence actually submitted? 2 procedure families (lumbar-MRI imaging, "
            "PT/rehab), 10 requirements paraphrased from real CMS LCD policy text, 150 synthetic "
            "cases (50 per outcome) with an injected, hand-checked ground truth — a controlled "
            "test bed, not a claim of production accuracy.")

    cases = load_evidence_cases()
    policy_by_id = {p["policy_id"]: p for p in ALL_POLICIES}
    families = sorted(policy_by_id, key=lambda pid: policy_by_id[pid]["family"])

    pick1, pick2, pick3 = st.columns(3)
    policy_id = pick1.selectbox(
        "Procedure type", families,
        format_func=lambda pid: FAMILY_LABEL.get(policy_by_id[pid]["family"],
                                                 policy_by_id[pid]["family"].replace("_", " ").title()))
    reqs = sorted({c["req_id"] for c in cases if c["policy_id"] == policy_id})
    req_id = pick2.selectbox("Insurer requirement", reqs)
    variants = [c for c in cases if c["policy_id"] == policy_id and c["req_id"] == req_id]
    gold = pick3.selectbox("Example case", [c["gold_variant"] for c in variants],
                           format_func=lambda v: v.capitalize(),
                           help="A synthetic case with a known, injected correct answer.")
    case = next(c for c in variants if c["gold_variant"] == gold)
    tfidf_result = tfidf_classify(case)
    cache_before = semantic_cache_size()
    semantic_result = semantic_classify(case)
    new_embeddings = semantic_cache_size() - cache_before
    report = explain_case(case)

    st.caption(f"{policy_by_id[policy_id]['source']}  ·  “{case['requirement_text']}”")

    STAGE_H = 430
    stage1, stage2, stage3 = st.columns(3, gap="medium")

    with stage1:
        st.markdown("**① What was submitted**")
        with st.container(border=True, height=STAGE_H):
            st.caption("Documentation included with the claim")
            for chunk in case["submitted_chunks"]:
                st.markdown(f"> {chunk}")
            with st.expander("Full patient record (incl. what was withheld)"):
                for chunk in case["full_record_chunks"]:
                    tag = "✓ submitted" if chunk in case["submitted_chunks"] else "not submitted"
                    st.markdown(f"- {chunk}  \n  &nbsp;&nbsp;*({tag})*")

    with stage2:
        st.markdown("**② What the system found**")
        with st.container(border=True, height=STAGE_H):
            if report.cited_chunk:
                st.success(f"Found matching evidence:\n\n“{report.cited_chunk}”")
            else:
                st.error("No evidence in the record matched the requirement.")
            st.caption(f"Match confidence {report.confidence:.2f} "
                       f"(semantic similarity to the requirement).")
            with st.expander("Details — retrieval scores (TF-IDF vs. semantic)"):
                if new_embeddings > 0:
                    st.caption(f"🔄 computed {new_embeddings} new embedding(s) just now "
                               f"({semantic_cache_size()} cached).")
                else:
                    st.caption(f"⚡ served from cache ({semantic_cache_size()} cached).")
                score_df = pd.DataFrame([
                    {"method": "TF-IDF", "submitted": round(tfidf_result["sub_score"], 2),
                     "record": round(tfidf_result["rec_score"], 2)},
                    {"method": "Semantic", "submitted": round(semantic_result["sub_score"], 2),
                     "record": round(semantic_result["rec_score"], 2)},
                ])
                st.dataframe(score_df, use_container_width=True, hide_index=True)

    with stage3:
        st.markdown("**③ Verdict**")
        with st.container(border=True, height=STAGE_H):
            color = STATUS_COLOR[report.status]
            st.markdown(
                f"<div style='padding:12px;border-radius:10px;background:{color}18;"
                f"border:1px solid {color};text-align:center'>"
                f"<span style='font-size:1.5rem;font-weight:800;color:{color}'>"
                f"{STATUS_LABEL[report.status]}</span></div>", unsafe_allow_html=True)
            match = "matches" if report.status == case["gold_variant"] else "differs from"
            st.caption(f"{match} the known correct answer ({case['gold_variant']}).")
            st.markdown(f"**What to do:** {report.suggested_action}")
            with st.expander("Full explanation"):
                st.write(report.rationale)

    st.divider()
    st.markdown("**How often is it right? (150 test cases)**")
    with open(ART / "ablation_summary.json") as f:
        summary = json.load(f)
    r1, r2 = st.columns(2)
    r1.metric("TF-IDF — keyword match", f"{summary['tfidf_baseline_accuracy']:.1%}")
    r2.metric("Semantic — meaning match", f"{summary['semantic_method_accuracy']:.1%}")
    st.caption("Semantic retrieval catches paraphrased evidence that keyword matching misses.")
    with st.expander("Detailed results — confusion matrices & score separation"):
        fig_cols = st.columns(3)
        for i, (fname, caption) in enumerate([
            ("evidence_confusion_tfidf.png", "TF-IDF confusion matrix"),
            ("evidence_confusion_semantic.png", "Semantic confusion matrix"),
            ("evidence_score_distribution.png", "Score separation (semantic)")]):
            p = FIG / fname
            if p.exists():
                fig_cols[i].image(str(p), caption=caption, use_column_width=True)


def main():
    _inject_css()
    st.title("🛡️ DenialGuard")
    st.markdown("#### Catch insurance-claim denials *before* you submit")
    st.write("Enter a claim and DenialGuard estimates how likely it is to be **denied**, "
             "explains **why**, and suggests **what to fix** — plus a separate check on whether "
             "the supporting documentation is complete.")

    if not (ART / "bundle.pkl").exists():
        st.error("Artifacts not found. Run:  `python scripts/run_phase4.py`")
        return
    art, test = load_artifacts()

    t1, t2, t3 = st.tabs(["🔎 Check a claim", "📊 How well it works",
                          "📄 Check the documentation"])
    with t1:
        score_tab(art, test)
    with t2:
        results_tab(art)
    with t3:
        evidence_tab()

    st.divider()
    st.caption(f"CS6140 · Phase 4 (retrieval-augmented denial prediction) · "
               f"{len(art.retriever):,} historical claims indexed · {art.retriever.backend} backend")


if __name__ == "__main__":
    main()
