"""
Shared evaluation harness.

WHY: the proposal's whole point is a cross-phase ablation study — Phase 1's
AUROC=0.5 baseline vs Phase 2's XGBoost vs Phase 4's retrieval-augmented model.
That comparison is only valid if every phase computes AUROC/F1 the same way
(same averaging, same threshold selection, same CV strategy). This module is
the single place that logic lives, and every phase's train script imports it
and logs results to MLflow with the same metric names — that's what makes
the final ablation table possible without a manual reconciliation pass.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

try:
    import mlflow
except ImportError:  # mlflow optional at import time for lightweight test envs
    mlflow = None

from shared.config.settings import settings


@dataclass
class EvalResult:
    phase: str
    model_name: str
    auroc: float
    f1: float
    precision: float
    recall: float
    n_samples: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    phase: str,
    model_name: str,
    threshold: float = 0.5,
) -> EvalResult:
    y_pred = (y_prob >= threshold).astype(int)
    return EvalResult(
        phase=phase,
        model_name=model_name,
        auroc=float(roc_auc_score(y_true, y_prob)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        n_samples=len(y_true),
    )


def log_to_mlflow(result: EvalResult, params: dict[str, Any] | None = None) -> None:
    """Every phase calls this identically so runs land in one comparable experiment."""
    if mlflow is None:
        raise ImportError("mlflow not installed — `pip install mlflow`")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name=f"{result.phase}-{result.model_name}"):
        mlflow.log_params(params or {})
        mlflow.log_params({"phase": result.phase, "model_name": result.model_name})
        mlflow.log_metrics(
            {
                "auroc": result.auroc,
                "f1": result.f1,
                "precision": result.precision,
                "recall": result.recall,
                "n_samples": result.n_samples,
            }
        )
