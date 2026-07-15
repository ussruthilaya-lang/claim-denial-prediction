# Phase 1 — Structured Baseline (owner: Teammate B)

Reproduces Hiremath et al. (IEEE, 2025): logistic regression + decision tree
on structured Kaggle claim fields, with SMOTE, backward elimination, and
stratified k-fold CV.

## Contract with the rest of the repo

- Input: raw Kaggle CSV → transform into `shared.schemas.claim.ClaimRecord`
  (see `shared/schemas/claim.py`). Keeping this contract means Phase 2 can
  reuse your preprocessing without rewriting it, and Phase 4 can retrieve
  from the same records.
- Output: log every run via `shared.utils.eval.evaluate()` +
  `log_to_mlflow()` — this is what makes your AUROC directly comparable to
  Phase 2/3/4 in the final ablation table without a manual reconciliation.

## Structure

```
phase1_baseline/
├── data/          # gitignored — put Kaggle CSV here locally
├── notebooks/      # exploratory work
├── src/            # preprocess.py, train.py — the reproducible pipeline
└── tests/
```

## Getting started

```bash
pip install -e ../shared
pip install -r requirements.txt
python src/train.py --data-path data/claims.csv
```

Known result to reproduce first (from the proposal's preliminary results):
LR with C=0.1 → ROC AUC ≈ 0.5. That's the number to beat in Phase 2.
