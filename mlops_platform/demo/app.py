"""
DenialGuard — Phase 4 decision-support demo (the "working application").

Three tabs:
  1. Score a claim — pick or build a claim, get a denial-risk band, the top-k
     most similar historical claims, a plain-language rationale, and a suggested
     action. This is retrieval-augmented denial PREVENTION, not just scoring.
  2. Model results — the ablation, calibration, cost, SHAP, label-noise, and
     harmonization figures, so the same app carries the presentation's Results.
  3. Evidence completeness (extension) — a separate, explicitly scoped test:
     given one CMS payer requirement and one synthetic patient record, is the
     requirement supported, supported-but-omitted, or genuinely unsupported?
     Every retrieval step is shown, not just the final answer, so the tab
     reads as a transparent test of the method rather than a black box.

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

STATUS_COLOR = {"complete": "#0ca30c", "omitted": "#fab219", "unsupported": "#d03b3b"}

ART = ROOT / "phase4_rag_agentic" / "artifacts"
FIG = ART / "figures"

BAND_COLOR = {"HIGH": "#d03b3b", "ELEVATED": "#fab219", "LOW": "#0ca30c"}

st.set_page_config(page_title="DenialGuard · Phase 4", page_icon="🛡️", layout="wide")


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


def score_tab(art, test):
    st.subheader("Score a claim for denial risk")
    left, right = st.columns([1, 1.4])

    with left:
        mode = st.radio("Claim source", ["Pick a sample claim", "Build a claim"],
                        horizontal=True)
        if mode == "Pick a sample claim":
            sample = test.sort_values("prob_augmented", ascending=False).head(200)
            options = sample["claim_id"].tolist()
            cid = st.selectbox("Claim (sorted by model risk)", options)
            row = test[test["claim_id"] == cid].iloc[0]
            claim = row_to_claim(row)
        else:
            payer = st.selectbox("Insurance", art.encoder.payers)
            cpt = st.selectbox("CPT (procedure)",
                               [f"{c} — {PROCEDURE_BY_CPT[c].label}" for c in art.encoder.cpts])
            cpt = cpt.split(" — ")[0]
            icd = st.selectbox("ICD-10 (diagnosis)", _icd_options())
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

        with st.container(border=True):
            st.markdown("**Input — claim record**")
            filing_days = (claim.submission_date - claim.service_date).days
            fields = pd.DataFrame([
                {"field": "Claim ID", "value": claim.claim_id},
                {"field": "Provider", "value": claim.provider_id},
                {"field": "ICD-10 (diagnosis)", "value": claim.icd10_code},
                {"field": "CPT (procedure)",
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

        go = st.button("Assess denial risk", type="primary")

    with right:
        if not go:
            st.info("Choose or build a claim, then **Assess denial risk**.")
            return
        ds = explain_claim(art, claim)
        color = BAND_COLOR[ds.risk_band]
        st.markdown(
            f"<div style='padding:16px;border-radius:10px;background:{color}22;"
            f"border:1px solid {color}'>"
            f"<span style='font-size:2.4rem;font-weight:700;color:{color}'>"
            f"{ds.denial_probability:.0%}</span> "
            f"<span style='font-size:1.1rem;color:{color};font-weight:600'>"
            f"denial risk · {ds.risk_band}</span></div>",
            unsafe_allow_html=True)
        st.markdown(f"**Suggested action:** {ds.suggested_action}")
        st.markdown(f"**Why:** {ds.rationale}")

        st.markdown("**Most similar past claims (retrieved evidence)**")
        ev = pd.DataFrame([{
            "claim_id": e.claim.claim_id, "similarity": round(e.similarity, 3),
            "provider": e.claim.provider_id, "payer": e.claim.insurance_type,
            "cpt": e.claim.cpt_code, "billed": e.claim.billed_amount,
            "outcome": "DENIED" if e.claim.denied else "paid",
            "reason": e.claim.reason_code or "",
        } for e in ds.evidence])
        st.dataframe(ev, use_container_width=True, hide_index=True)
        if ds.top_reasons:
            st.caption("Denial reasons in this neighbourhood: "
                       + " · ".join(f"{r} ({n})" for r, n in ds.top_reasons))


def results_tab(art):
    st.subheader("Phase 4 results")
    cop = art.cost_operating_point
    c1, c2, c3 = st.columns(3)
    c1.metric("Cost-optimal threshold", f"{cop['threshold']:.2f}")
    c2.metric("Cost per claim", f"${cop['cost_per_claim']:.2f}")
    c3.metric("Savings vs do-nothing", f"${cop['savings_vs_do_nothing']:,.0f}")

    figs = [("ablation_auroc.png", "Ablation: retrieval lift, ceiling, leakage trap"),
            ("shap_bar.png", "Feature importance (green = retrieval features)"),
            ("calibration.png", "Calibration"),
            ("cost_curve.png", "Cost-sensitive operating point"),
            ("noise_sweep.png", "Label-noise robustness"),
            ("harmonization_psi.png", "Population stability (train vs test)")]
    cols = st.columns(2)
    for i, (fname, caption) in enumerate(figs):
        p = FIG / fname
        if p.exists():
            cols[i % 2].image(str(p), caption=caption, use_column_width=True)


@st.cache_resource
def load_evidence_cases():
    cases = generate_cases()
    build_global_index(cases)  # fits the TF-IDF vectorizer once, globally
    return cases


def evidence_tab():
    st.subheader("Evidence completeness — Phase 4 extension")
    with st.expander("About this test (scope, 30 seconds)"):
        st.write(
            "A separate, narrower test from the Phase 1-4 denial model in "
            "the other tabs: for one specific payer requirement, is it "
            "supported by the patient's record — and if so, was that "
            "evidence actually submitted? 2 procedure families (lumbar-MRI "
            "imaging, PT/rehab), 10 requirements paraphrased from real CMS "
            "LCD policy text, 150 synthetic cases (50 per outcome) with an "
            "injected, hand-checked ground truth — a controlled test bed, "
            "not a claim of production accuracy.")

    cases = load_evidence_cases()
    policy_by_id = {p["policy_id"]: p for p in ALL_POLICIES}
    families = sorted(policy_by_id, key=lambda pid: policy_by_id[pid]["family"])

    pick1, pick2, pick3 = st.columns(3)
    policy_id = pick1.selectbox("Procedure family", families,
                                format_func=lambda pid: policy_by_id[pid]["family"])
    reqs = sorted({c["req_id"] for c in cases if c["policy_id"] == policy_id})
    req_id = pick2.selectbox("Requirement", reqs)
    variants = [c for c in cases if c["policy_id"] == policy_id and c["req_id"] == req_id]
    gold = pick3.selectbox("Case to run", [c["gold_variant"] for c in variants],
                           help="A specific synthetic case with known, injected ground truth.")
    case = next(c for c in variants if c["gold_variant"] == gold)
    tfidf_result = tfidf_classify(case)
    cache_before = semantic_cache_size()
    semantic_result = semantic_classify(case)
    new_embeddings = semantic_cache_size() - cache_before
    report = explain_case(case)

    st.caption(f"{policy_by_id[policy_id]['source']}  ·  “{case['requirement_text']}”")

    STAGE_H = 420
    stage1, stage2, stage3 = st.columns(3, gap="medium")

    with stage1:
        st.markdown("**① Input** — patient record")
        with st.container(border=True, height=STAGE_H):
            st.caption("Submitted with claim")
            for chunk in case["submitted_chunks"]:
                st.markdown(f"> {chunk}")
            with st.expander("Full record (incl. what was withheld)"):
                for chunk in case["full_record_chunks"]:
                    tag = "✓ submitted" if chunk in case["submitted_chunks"] else "not submitted"
                    st.markdown(f"- {chunk}  \n  &nbsp;&nbsp;*({tag})*")

    with stage2:
        st.markdown("**② Retrieve** — top-matching chunk")
        with st.container(border=True, height=STAGE_H):
            if report.cited_chunk:
                st.success(f"“{report.cited_chunk}”")
            else:
                st.error("No chunk in the record cleared the match threshold.")
            st.caption(f"Semantic search · cosine similarity {report.confidence:.2f} "
                       f"vs. the requirement — this chunk is passed forward to generate the output.")
            if new_embeddings > 0:
                st.caption(f"🔄 computed {new_embeddings} new embedding(s) just now "
                           f"({semantic_cache_size()} cached total)")
            else:
                st.caption(f"⚡ served from cache — already embedded earlier "
                           f"this session ({semantic_cache_size()} cached total)")
            score_df = pd.DataFrame([
                {"method": "TF-IDF", "submitted": round(tfidf_result["sub_score"], 2),
                 "record": round(tfidf_result["rec_score"], 2)},
                {"method": "Semantic", "submitted": round(semantic_result["sub_score"], 2),
                 "record": round(semantic_result["rec_score"], 2)},
            ])
            st.dataframe(score_df, use_container_width=True, hide_index=True)

    with stage3:
        st.markdown("**③ Generate** — decision support")
        with st.container(border=True, height=STAGE_H):
            color = STATUS_COLOR[report.status]
            st.markdown(
                f"<div style='padding:12px;border-radius:8px;background:{color}22;"
                f"border:1px solid {color};text-align:center'>"
                f"<span style='font-size:1.6rem;font-weight:700;color:{color}'>"
                f"{report.status.upper()}</span></div>", unsafe_allow_html=True)
            match = "matches" if report.status == case["gold_variant"] else "differs from"
            st.caption(f"{match} known ground truth ({case['gold_variant']})")
            st.markdown(f"**Action:** {report.suggested_action}")
            with st.expander("Full explanation"):
                st.write(report.rationale)

    st.divider()
    st.markdown("#### Results — accuracy over all 150 cases")
    with open(ART / "ablation_summary.json") as f:
        summary = json.load(f)
    r1, r2 = st.columns(2)
    r1.metric("TF-IDF baseline accuracy", f"{summary['tfidf_baseline_accuracy']:.1%}")
    r2.metric("Semantic method accuracy", f"{summary['semantic_method_accuracy']:.1%}")
    fig_cols = st.columns(3)
    for i, (fname, caption) in enumerate([
        ("evidence_confusion_tfidf.png", "TF-IDF confusion matrix"),
        ("evidence_confusion_semantic.png", "Semantic confusion matrix"),
        ("evidence_score_distribution.png", "Score separation (semantic)")]):
        p = FIG / fname
        if p.exists():
            fig_cols[i].image(str(p), caption=caption, use_column_width=True)


def main():
    st.title("🛡️ DenialGuard — retrieval-augmented denial prevention")
    st.caption("Phase 4 · CS6140 · surfaces similar past denied claims and an "
               "actionable rationale, not just a probability.")
    if not (ART / "bundle.pkl").exists():
        st.error("Artifacts not found. Run:  `python scripts/run_phase4.py`")
        return
    art, test = load_artifacts()
    st.caption(f"Index backend: **{art.retriever.backend}** · "
               f"{len(art.retriever):,} historical claims indexed")
    t1, t2, t3 = st.tabs(["Score a claim", "Model results",
                         "Evidence completeness (extension)"])
    with t1:
        score_tab(art, test)
    with t2:
        results_tab(art)
    with t3:
        evidence_tab()


if __name__ == "__main__":
    main()
