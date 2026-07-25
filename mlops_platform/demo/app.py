"""
DenialGuard — Phase 4 decision-support demo (the "working application").

Plain-language, task-oriented UI with progressive disclosure: each tab leads
with the answer a non-expert needs, and pushes the ML detail (SHAP,
calibration, PSI, similar-claims table, retrieval scores) behind expanders so
a technical reviewer can still drill in.

Three tabs:
  1. Check a claim        — denial risk + why + suggested fix (retrieval-augmented).
  2. How well it works    — accuracy / cost / savings, with full diagnostics on demand.
  3. Required documentation — the evidence-completeness extension: is a payer
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

from phase4_rag_agentic.src.data_gen import (_DX_PHRASES, ICD10,
                                             PROCEDURE_BY_CPT, row_to_claim)
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
# Cross-phase ladder (structured -> +text -> +retrieval -> +both); written by
# scripts/run_all_phases.py to the repo-level artifacts/ dir.
UNIFIED_FIG = ROOT / "artifacts" / "unified_ablation.png"

# Claim-level risk bands (Score a claim).
BAND_COLOR = {"HIGH": "#d03b3b", "ELEVATED": "#fab219", "LOW": "#0ca30c"}
# Evidence-completeness statuses (Required documentation).
STATUS_COLOR = {"complete": "#0ca30c", "omitted": "#fab219", "unsupported": "#d03b3b"}
STATUS_LABEL = {"complete": "Documentation complete",
                "omitted": "Evidence omitted",
                "unsupported": "Not supported"}
FAMILY_LABEL = {"advanced_imaging": "Advanced imaging", "pt_rehab": "PT / rehab"}

st.set_page_config(page_title="DenialGuard", page_icon="🛡️", layout="wide")


def _inject_css():
    # Minimal declutter: hide only the Deploy button + footer. Keep the top-right
    # "⋮" menu so the built-in Light/Dark theme switcher (Settings → Theme) stays.
    st.markdown(
        "<style>"
        "[data-testid='stDeployButton']{display:none;}"
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


def _clean_dx_label(phrase: str) -> str:
    for article in ("an ", "a "):
        if phrase.startswith(article):
            phrase = phrase[len(article):]
            break
    return phrase[0].upper() + phrase[1:]


# ICD-10 code -> human-readable diagnosis name, built from the same
# category-ordered phrases data_gen.py uses to write clinical notes (so the
# label always matches what the note actually says, no separate hardcoded list).
_ICD_LABELS = {
    code: _clean_dx_label(phrase)
    for cat, codes in ICD10.items()
    for code, phrase in zip(codes, _DX_PHRASES[cat])
}

# Requirement id (e.g. "PT-1", "IMG-1") -> its actual CMS LCD requirement
# sentence, so the "Insurer requirement" picker doesn't show a bare code with
# no indication of what it means.
_REQ_TEXT = {req["req_id"]: req["text"]
            for policy in ALL_POLICIES for req in policy["requirements"]}


def _short_req_label(req_id: str, max_len: int = 55) -> str:
    text = _REQ_TEXT[req_id]
    return f"{req_id} — {text[:max_len]}{'…' if len(text) > max_len else ''}"


# ───────────────────────────── Tab 1: Check a claim ─────────────────────────
def score_tab(art, test):
    st.subheader("Score a claim for denial risk")
    st.markdown(
        "**What this does:** checks one insurance claim and estimates the "
        "chance it gets denied, using past claims that looked similar.\n\n"
        "**How to try it:**\n"
        "1. On the left, leave **\"Pick a sample claim\"** selected and choose "
        "any claim from the dropdown (they're sorted highest-risk first).\n"
        "2. Click the **Assess denial risk** button.\n"
        "3. The result appears on the right: a risk %, a plain-language reason, "
        "a suggested action, and the past claims it was compared against.\n\n"
        "(Or switch to **\"Build a claim\"** to type in your own claim details "
        "instead of picking an existing one.)"
    )
    left, right = st.columns([1, 1.4], gap="large")

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
            icd = st.selectbox("ICD-10 (diagnosis)",
                               [f"{c} — {_ICD_LABELS[c]}" for c in _icd_options()])
            icd = icd.split(" — ")[0]
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

    with st.expander("How is the cost calculated?"):
        st.markdown(
            "The review cutoff is set to **minimize real dollars**, not just error rate. "
            "Two mistakes are priced:\n\n"
            "- **Missed denial** — flagged as safe but actually denied: **\\$400**\n"
            "- **Unnecessary review** — flagged but would've been paid: **\\$40**\n\n"
            "Every cutoff is swept, and the one with the lowest "
            "**\\$400 × missed denials + \\$40 × unnecessary reviews** (on the held-out test "
            "set) is chosen. Because a missed denial costs 10× a wasted review, the best "
            "cutoff is low — it pays to flag aggressively. **Savings vs. no model** compares "
            "that cost against reviewing nothing, where every denial is missed.\n\n"
            "_The \\$400 / \\$40 are illustrative cost assumptions, not billed amounts._")

    st.divider()
    with st.expander("🔍 Check this to understand more — how each phase of the project adds up"):
        if UNIFIED_FIG.exists():
            chart, notes = st.columns([3, 2], gap="large")
            chart.image(str(UNIFIED_FIG), use_column_width=True)
            notes.markdown(
                "One linked dataset, four models — **higher AUROC = better**:\n\n"
                "- **Structured billing data** (Phases 1–2): **0.733** — the baseline\n"
                "- **+ clinical note** (Phase 3): **0.769** — the note adds signal the form misses\n"
                "- **+ retrieval** over past claims (Phase 4): **0.755**\n"
                "- **Both together**: **0.795** — best\n\n"
                "The dashed line is the **oracle ceiling (0.883)** — the most any model "
                "could recover on this data.")
            notes.caption("The live scorer in **Check a claim** is the structured + "
                          "retrieval (Phase 4) model.")
        else:
            st.info("Run `python scripts/run_all_phases.py` to generate the cross-phase figure.")

    with st.expander("Technical detail — Phase 4 diagnostics (for reviewers)"):
        figs = [("ablation_auroc.png",
                 "Phase 4 ablation — retrieval lift, oracle ceiling, and the leaky-index trap"),
                ("shap_bar.png", "What drives the prediction (green = retrieval features)"),
                ("calibration.png", "Calibration — predicted vs. actual denial rate"),
                ("cost_curve.png", "Cost-sensitive operating point"),
                ("noise_sweep.png", "Robustness when labels are noisy"),
                ("harmonization_psi.png", "Population stability, train vs. test (PSI)")]
        cols = st.columns(2)
        for i, (fname, caption) in enumerate(figs):
            fp = FIG / fname
            if fp.exists():
                cols[i % 2].image(str(fp), caption=caption, use_column_width=True)


# ─────────────────────── Tab 3: Required documentation ──────────────────────
def load_evidence_cases():
    cases = generate_cases()
    build_global_index(cases)  # fits the TF-IDF vectorizer once, globally
    return cases


def evidence_tab():
    st.markdown(
        "**Why this matters:** payers don't only deny claims for lack of medical "
        "necessity — many denials happen because a specific documentation requirement "
        "wasn't met (e.g. *\"6 weeks of conservative therapy must be documented\"* before "
        "an MRI is approved). Even when that evidence exists in the chart, the claim "
        "still gets denied if it wasn't included in what was submitted. This check "
        "catches that specific, avoidable failure *before* submission, one requirement "
        "at a time: does the evidence exist, and was it actually sent with the claim?\n\n"
        "**How this differs from \"Check a claim\":** that tab scores a whole claim's "
        "overall denial risk as a percentage. This one zooms into a single documentation "
        "requirement and gives a definitive answer — complete, evidence omitted, or "
        "evidence unsupported — rather than a probability."
    )
    st.caption(
        "Built from 2 real CMS LCD policy families (lumbar-MRI imaging, PT/rehab) with "
        "10 requirements paraphrased from actual CMS documentation standards, scored "
        "against 150 synthetic cases that each have a known, hand-injected correct "
        "answer — a controlled test bed comparing two ways of finding evidence in a "
        "chart (keyword search vs. meaning-based search), not a claim of production "
        "accuracy."
    )
    st.markdown(
        "**How to browse the 150 test cases:**\n"
        "1. Pick a **Procedure type** — the 2 CMS policies this test bed covers "
        "(\"Advanced imaging\" = lumbar MRI, \"PT / rehab\" = physical therapy).\n"
        "2. Pick an **Insurer requirement** — each policy has 5, numbered 1st through 5th "
        "(\"IMG-1\"..\"IMG-5\" for imaging, \"PT-1\"..\"PT-5\" for PT/rehab); the dropdown "
        "shows what each one actually requires.\n"
        "3. Pick an **Example case** — 5 synthetic patients × 3 outcomes each "
        "(Complete / Omitted / Unsupported), each with a known correct answer.\n"
        "4. Read the three panels below: what was actually submitted, what the system "
        "found in the full record, and whether its verdict matches the known answer."
    )

    st.markdown("#### Browse the test cases")
    cases = load_evidence_cases()
    policy_by_id = {p["policy_id"]: p for p in ALL_POLICIES}
    families = sorted(policy_by_id, key=lambda pid: policy_by_id[pid]["family"])

    pick1, pick2, pick3 = st.columns(3)
    policy_id = pick1.selectbox(
        "Procedure type", families,
        format_func=lambda pid: FAMILY_LABEL.get(policy_by_id[pid]["family"],
                                                 policy_by_id[pid]["family"].replace("_", " ").title()))
    reqs = sorted({c["req_id"] for c in cases if c["policy_id"] == policy_id})
    req_id = pick2.selectbox("Insurer requirement", reqs, format_func=_short_req_label)
    # 5 synthetic patients x 3 outcomes each for this requirement, in that fixed
    # order (see evidence_data_gen.generate_cases) — label each uniquely (patient
    # + outcome) instead of by outcome alone, which repeated 5x and made every
    # instance of "Complete"/"Omitted"/"Unsupported" indistinguishable.
    variants = [c for c in cases if c["policy_id"] == policy_id and c["req_id"] == req_id]
    labels = [f"Patient {i // 3 + 1} — {c['gold_variant'].capitalize()}"
             for i, c in enumerate(variants)]
    idx = pick3.selectbox("Example case", range(len(variants)),
                         format_func=lambda i: labels[i],
                         help="Each patient has 3 versions of this requirement — a synthetic "
                              "case with a known, injected correct answer.")
    case = variants[idx]
    tfidf_result = tfidf_classify(case)
    cache_before = semantic_cache_size()
    semantic_result = semantic_classify(case)
    new_embeddings = semantic_cache_size() - cache_before
    report = explain_case(case)

    st.caption(f"{policy_by_id[policy_id]['source']}  ·  “{case['requirement_text']}”")

    STAGE_H = 380
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
    st.markdown("### How often is it right? (150 test cases)")
    with open(ART / "ablation_summary.json") as f:
        summary = json.load(f)
    r1, r2 = st.columns(2)
    r1.metric("TF-IDF — keyword match", f"{summary['tfidf_baseline_accuracy']:.1%}")
    r2.metric("Semantic — meaning match", f"{summary['semantic_method_accuracy']:.1%}")
    st.caption("Semantic search (matches by *meaning*) catches paraphrased evidence that exact "
              "keyword matching misses.")

    with st.expander("🔍 Check this to understand more — see exactly where each method "
                     "gets it right or wrong"):
        c1, c2 = st.columns(2)
        with c1:
            st.image(str(FIG / "evidence_confusion_tfidf.png"), use_column_width=True)
            st.caption(
                "Rows = the true answer, columns = what the method predicted — the diagonal "
                "is correct. TF-IDF is cautious: it never wrongly claims evidence exists (the "
                "bottom row, true 'unsupported', lands entirely in the 'unsupported' column). "
                "But it misses 30 of the 100 cases where evidence really was in the chart, "
                "calling them 'unsupported' just because the wording doesn't share enough "
                "exact keywords."
            )
        with c2:
            st.image(str(FIG / "evidence_confusion_semantic.png"), use_column_width=True)
            st.caption(
                "Same layout. Semantic search finds more of the real evidence (78 of 100 vs. "
                "TF-IDF's 70) because it matches by meaning, not exact wording — but that "
                "costs a little precision: in 4 cases it sees evidence that isn't actually "
                "there (the small off-diagonal counts in the 'complete'/'omitted' columns), "
                "something TF-IDF never does."
            )

        score_col, _ = st.columns([1, 1])
        with score_col:
            st.image(str(FIG / "evidence_score_distribution.png"), use_column_width=True)
            st.caption(
                "Why semantic search still makes mistakes: each case's best-match similarity "
                "score, split by whether evidence actually existed (green) or not (red). The "
                "two mostly separate — red mostly below ~0.35, green mostly above ~0.45 — but "
                "they overlap in between. Cases that fall in that overlap band are exactly "
                "the errors in the confusion matrices above: no single cutoff score sorts "
                "every case correctly."
            )


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

    t1, t2, t3 = st.tabs(["🔎 Check a claim", "📄 Required documentation",
                          "📊 How well it works"])
    with t1:
        score_tab(art, test)
    with t2:
        evidence_tab()
    with t3:
        results_tab(art)

    st.divider()
    st.caption(f"CS6140 · Phase 4 (retrieval-augmented denial prediction) · "
               f"{len(art.retriever):,} historical claims indexed · {art.retriever.backend} backend")


if __name__ == "__main__":
    main()
