# Phase 2 — GBM + SHAP (owner: Teammate B)

Replaces Phase 1's LR/DT with XGBoost + LightGBM, adds SHAP attribution per
claim.

## Contract with the rest of the repo

Same `ClaimRecord` input contract as Phase 1. Log via
`shared.utils.eval.evaluate()` / `log_to_mlflow()` with `phase="phase2_gbm_shap"`
so results land in the same MLflow experiment as every other phase.

SHAP values per claim should be persisted (e.g. to `data/shap_values.parquet`)
— Phase 4's rationale generation can reference "top SHAP features" as
additional evidence alongside retrieved similar claims, once both are stable.

## Structure

```
phase2_gbm_shap/
├── data/
├── notebooks/
├── src/           # train.py, shap_analysis.py
└── tests/
```
