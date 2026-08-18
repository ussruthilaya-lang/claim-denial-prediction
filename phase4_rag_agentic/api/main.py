"""
FastAPI serving layer for Phase 4's retrieval-augmented denial model.

WHY THIS EXISTS: the model and demo were already real and reproducible, but
"reproducible in a notebook" and "callable as a service" are different claims.
This is the smallest honest version of the second one: one prediction
endpoint wrapping the exact same `explain_claim` decision-support path the
Streamlit demo uses (so the API and the demo can never silently disagree),
a health check, and real request-count/latency metrics — not a mocked
monitoring section, an actual `/metrics` endpoint a Prometheus scraper could
hit.

Run locally:      uvicorn phase4_rag_agentic.api.main:app --reload
Run in Docker:    see phase4_rag_agentic/Dockerfile
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from phase4_rag_agentic.src.features import RetrievalFeaturizer
from phase4_rag_agentic.src.llm_demo import explain_claim
from phase4_rag_agentic.src.retriever import ClaimRetriever, RetrievedClaim
from shared.schemas.claim import ClaimRecord

ART = ROOT / "phase4_rag_agentic" / "artifacts"

REQUEST_COUNT = Counter(
    "phase4_requests_total", "Total requests served, by endpoint and status",
    ["endpoint", "status"])
REQUEST_LATENCY = Histogram(
    "phase4_request_latency_seconds", "Request latency in seconds", ["endpoint"])


class NeighbourClaim(BaseModel):
    claim_id: str
    similarity: float
    provider_id: str
    denied: bool | None
    reason_code: str | None


class PredictResponse(BaseModel):
    claim_id: str
    denial_probability: float
    risk_band: str
    suggested_action: str
    rationale: str
    neighbour_denial_rate: float
    similar_past_claims: list[NeighbourClaim]


_ARTIFACTS: SimpleNamespace | None = None


def _load_artifacts() -> SimpleNamespace:
    """Loads the pretrained Phase 4 bundle once, at process startup — the API
    never trains anything, it only serves an already-validated model, same
    as the demo app."""
    with open(ART / "bundle.pkl", "rb") as f:
        bundle = pickle.load(f)
    retriever = ClaimRetriever.load(ART / "claims_index.faiss")
    return SimpleNamespace(
        encoder=bundle["encoder"],
        model_augmented=bundle["model_augmented"],
        retriever=retriever,
        featurizer=RetrievalFeaturizer(retriever, k=bundle["k"]),
        cost_operating_point=bundle["cost_operating_point"],
    )


app = FastAPI(
    title="Phase 4 — Retrieval-Augmented Denial Prediction",
    description="Scores a claim and returns a grounded, cited rationale, "
                "not just a bare probability.",
    version="1.0.0",
)


@app.on_event("startup")
def _startup() -> None:
    global _ARTIFACTS
    _ARTIFACTS = _load_artifacts()


@app.middleware("http")
async def _metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    endpoint = request.url.path
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(elapsed)
    REQUEST_COUNT.labels(endpoint=endpoint, status=response.status_code).inc()
    return response


@app.get("/health")
def health() -> dict:
    """Liveness/readiness check: reports whether the model artifacts are
    actually loaded, not just whether the process is running."""
    loaded = _ARTIFACTS is not None
    return {
        "status": "ok" if loaded else "not_ready",
        "model_loaded": loaded,
        "retrieval_backend": _ARTIFACTS.retriever.backend if loaded else None,
        "index_size": len(_ARTIFACTS.retriever) if loaded else 0,
    }


@app.get("/metrics")
def metrics() -> Response:
    """Prometheus-scrapable request count + latency, real not illustrative —
    hit this a few times after calling /predict and the numbers move."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse)
def predict(claim: ClaimRecord) -> PredictResponse:
    """Scores one claim: retrieves similar historical claims, returns a
    calibrated denial probability plus a rationale grounded in that
    retrieved evidence — the same decision-support path the demo uses."""
    ds = explain_claim(_ARTIFACTS, claim)
    return PredictResponse(
        claim_id=ds.claim_id,
        denial_probability=ds.denial_probability,
        risk_band=ds.risk_band,
        suggested_action=ds.suggested_action,
        rationale=ds.rationale,
        neighbour_denial_rate=ds.neighbour_denial_rate,
        similar_past_claims=[
            NeighbourClaim(
                claim_id=e.claim.claim_id, similarity=round(e.similarity, 3),
                provider_id=e.claim.provider_id, denied=e.claim.denied,
                reason_code=e.claim.reason_code,
            )
            for e in ds.evidence
        ],
    )
