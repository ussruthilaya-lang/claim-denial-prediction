"""
Serving layer: one API, model-agnostic.

WHY THIS EXISTS AS A SEPARATE LAYER (not "just call the model in a script"):
In production, the *serving contract* (what a client POSTs, what they get back)
should stay stable even as the model behind it changes from Phase 1's logistic
regression to Phase 4's retrieval-augmented pipeline. This file is deliberately
dumb about model internals — it asks the model registry "give me whatever is
tagged `settings.model_stage`" and calls `.predict()`. That's the difference
between "a notebook that outputs a number" and "a system a hospital's billing
software could actually call."

For Phase 4 specifically, `predict()` also returns the top-k retrieved similar
claims and the LLM's rationale — this is what makes it an *agentic* layer
rather than a bare classifier: the response is explainable by construction.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from shared.config.settings import settings
from shared.schemas.claim import ClaimRecord

logger = logging.getLogger("serving")

_model = None  # loaded once at startup, not per-request


class PredictionResponse(BaseModel):
    claim_id: str
    denial_probability: float
    denied_prediction: bool
    model_stage: str
    # populated only when the loaded model is the Phase 4 RAG layer
    similar_past_claims: list[str] | None = None
    rationale: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    _model = _load_current_model()
    logger.info("Loaded model for stage=%s", settings.model_stage)
    yield
    _model = None


app = FastAPI(title="Claim Denial Prediction API", lifespan=lifespan)


def _load_current_model():
    """
    Resolves the active model from the MLflow model registry by stage.
    Kept as a thin function (not inlined in lifespan) so it's independently
    testable and so swapping the resolution strategy (e.g. to a static
    artifact path during early dev, before anything is registered) is a
    one-function change.
    """
    import mlflow

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    try:
        return mlflow.pyfunc.load_model(f"models:/claim-denial/{settings.model_stage}")
    except Exception as e:  # registry empty during early dev — fail loud but don't crash import
        logger.warning("No registered model found yet (%s). API will 503 on /predict.", e)
        return None


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(claim: ClaimRecord):
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="No model registered yet under stage="
            f"'{settings.model_stage}'. Train and register a model first.",
        )

    prob = float(_model.predict([claim.model_dump()])[0])
    return PredictionResponse(
        claim_id=claim.claim_id,
        denial_probability=prob,
        denied_prediction=prob >= 0.5,
        model_stage=settings.model_stage,
    )
