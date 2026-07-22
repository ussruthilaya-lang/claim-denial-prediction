"""
WHY this test matters as a pattern: it's the smallest possible test that
proves the retriever's core contract (add vectors, get the nearest ones back
in similarity order) without needing FAISS's internals to be understood by
the reader. Every phase should have at least one test like this — cheap to
run, catches index/metadata drift immediately if someone changes `add()` or
`query()` later.
"""
from datetime import date

import numpy as np

# No faiss import-skip: the retriever falls back to an exact-cosine NumPy
# backend with identical query semantics, so this contract must hold on any
# machine, faiss installed or not.
from phase4_rag_agentic.src.retriever import ClaimRetriever
from shared.schemas.claim import ClaimRecord


def _claim(claim_id: str, denied: bool) -> ClaimRecord:
    return ClaimRecord(
        claim_id=claim_id,
        provider_id="prov-1",
        icd10_code="E11.9",
        cpt_code="99214",
        insurance_type="private",
        billed_amount=250.0,
        service_date=date(2026, 1, 1),
        submission_date=date(2026, 1, 3),
        denied=denied,
    )


def test_retriever_returns_nearest_neighbor_first():
    retriever = ClaimRetriever(embedding_dim=4)

    embeddings = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],  # orthogonal to query -> similarity 0
            [1.0, 0.0, 0.0, 0.0],  # nearly identical direction to query -> high similarity
        ]
    )
    claims = [_claim("far", denied=False), _claim("near", denied=True)]
    retriever.add(embeddings, claims)

    query = np.array([0.95, 0.05, 0.0, 0.0])
    results = retriever.query(query, k=2)

    assert results[0].claim.claim_id == "near"
    assert results[0].similarity >= results[1].similarity


def test_query_excludes_self_to_prevent_leakage():
    """The self-exclusion guard is the core leakage protection: a claim must
    never retrieve itself, or its own outcome leaks into its own features."""
    retriever = ClaimRetriever(embedding_dim=4)
    embeddings = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    claims = [_claim("self", denied=True), _claim("other", denied=False)]
    retriever.add(embeddings, claims)

    # Query is identical to "self" — without the guard it would rank first.
    results = retriever.query(embeddings[0], k=1, exclude_ids={"self"})

    assert len(results) == 1
    assert results[0].claim.claim_id == "other"
