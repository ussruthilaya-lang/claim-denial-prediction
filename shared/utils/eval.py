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


def _percentiles(values: list[float], ci: float) -> list[float]:
    lo, hi = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    return [float(np.percentile(values, lo)), float(np.percentile(values, hi))]


def bootstrap_auroc_lift(
    y_true: np.ndarray,
    p_base: np.ndarray,
    p_better: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
    ci: float = 0.95,
) -> dict[str, Any]:
    """PAIRED percentile bootstrap for an AUROC lift (better - base).

    A phase-vs-phase lift of a few AUROC points invites the question "is that
    real or noise?". This answers it: resample the test rows with replacement and
    score BOTH models on the SAME resample each iteration (their errors are
    correlated, so pairing gives the correct, tighter CI on the difference than
    two independent CIs would). Returns each model's CI, the lift CI, and a
    one-sided bootstrap p-value = fraction of resamples where the lift is <= 0.
    Reusable for any phase pair (text lift, retrieval lift, ...).
    """
    y_true = np.asarray(y_true)
    p_base = np.asarray(p_base)
    p_better = np.asarray(p_better)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    base, better, lift = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if yt.min() == yt.max():  # need both classes to define AUROC
            continue
        a = roc_auc_score(yt, p_base[idx])
        b = roc_auc_score(yt, p_better[idx])
        base.append(a)
        better.append(b)
        lift.append(b - a)
    lift_arr = np.asarray(lift)
    return {
        "auroc_base": float(roc_auc_score(y_true, p_base)),
        "auroc_better": float(roc_auc_score(y_true, p_better)),
        "auroc_base_ci": _percentiles(base, ci),
        "auroc_better_ci": _percentiles(better, ci),
        "lift_point": float(roc_auc_score(y_true, p_better) - roc_auc_score(y_true, p_base)),
        "lift_mean": float(lift_arr.mean()),
        "lift_ci": _percentiles(lift_arr, ci),
        "p_value_one_sided": float((lift_arr <= 0).mean()),
        "n_boot": int(len(lift_arr)),
    }
