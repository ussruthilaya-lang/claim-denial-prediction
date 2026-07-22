"""
DenialGuard — Phase 4 decision-support demo (the "working application").

Two tabs:
  1. Score a claim — pick or build a claim, get a denial-risk band, the top-k
     most similar historical claims, a plain-language rationale, and a suggested
     action. This is retrieval-augmented denial PREVENTION, not just scoring.
  2. Model results — the ablation, calibration, cost, SHAP, label-noise, and
     harmonization figures, so the same app carries the presentation's Results.

Run:  streamlit run mlops_platform/demo/app.py
Needs the Phase 4 artifacts:  python scripts/run_phase4.py   (writes them)
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from phase4_rag_agentic.src.data_gen import ICD10, PROCEDURE_BY_CPT, row_to_claim
from phase4_rag_agentic.src.features import RetrievalFeaturizer
from phase4_rag_agentic.src.llm_demo import explain_claim
from phase4_rag_agentic.src.retriever import ClaimRetriever

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
            cols[i % 2].image(str(p), caption=caption, use_container_width=True)


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
    t1, t2 = st.tabs(["Score a claim", "Model results"])
    with t1:
        score_tab(art, test)
    with t2:
        results_tab(art)


if __name__ == "__main__":
    main()
