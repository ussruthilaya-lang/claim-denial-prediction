# From Codes to Context: Claim Denial Prediction

CS6140 · Northeastern University · Summer 2026

A four-phase progressive pipeline for predicting medical claim denials — from
structured-data baselines to a retrieval-augmented, LLM-assisted decision layer.

## Why this repo is structured this way

Most student ML projects fail production-readiness for one reason: every phase
is a standalone notebook, nobody can run anyone else's code, and there's no
single place that says "here's how a claim actually flows through the system."

This repo is built like a real ML platform, not four disconnected notebooks:

- **Each phase is an installable Python package** (`phase1_baseline/`,
  `phase2_gbm_shap/`, `phase3_clinicalbert/`, `phase4_rag_agentic/`) with its
  own `src/`, `tests/`, and `requirements.txt`. This means Phase 2 can import
  Phase 1's preprocessing without copy-pasting code, and each teammate can
  `pip install -e .` only what they need.
- **`shared/`** holds the data schemas, config loading, and utility code that
  every phase depends on (e.g. the claim record schema, the AUROC/F1 eval
  harness). One source of truth prevents four slightly-different definitions
  of "what is a claim" from drifting apart.
- **`mlops_platform/`** is the MLOps layer that turns "four models" into "one
  system": Docker Compose for local dev, MLflow for experiment tracking so we
  can compare AUROC across all four phases in one place, a FastAPI serving
  layer that wraps whichever model is "current," and GitHub Actions CI so
  broken code never merges.
- **Phase ownership is explicit** (see below) but the platform layer is
  everyone's dependency — this is intentional. In production ML teams, the
  platform is the thing that outlives any one model.

## Team ownership

| Owner | Scope | Deliverable |
|---|---|---|
| **You (platform + Phase 4)** | `mlops_platform/`, `shared/`, `phase4_rag_agentic/` | MLOps infra (Docker, CI, MLflow, serving) + FAISS retrieval-augmented layer + GPT-4 zero-shot baseline |
| Teammate B | `phase1_baseline/`, `phase2_gbm_shap/` | LR/DT baseline reproduction + XGBoost/LightGBM + SHAP |
| Teammate C | `phase3_clinicalbert/` | ClinicalBERT embeddings over MIMIC-IV-Note, fused with structured features |

Every phase reads from `shared/schemas` and writes metrics through
`shared/utils/eval.py` so results are directly comparable in the final
ablation study — that comparability is the whole point of the "progressive"
design in the proposal.

## Quickstart

```bash
cp .env.example .env
make setup          # creates venv, installs shared + platform deps
make up              # docker compose up: mlflow, postgres (feature store), api
make test            # runs pytest across all phase packages
```

See `docs/architecture/` for system diagrams and `docs/adr/` for decisions
(e.g. why FAISS over pgvector, why GCP over AWS) with the reasoning kept
next to the decision — useful for you to point to in interviews.

## Repo layout

```
.
├── phase1_baseline/       # Teammate B — LR + DT, SMOTE, stratified k-fold
├── phase2_gbm_shap/       # Teammate B — XGBoost/LightGBM + SHAP
├── phase3_clinicalbert/   # Teammate C — ClinicalBERT + MIMIC-IV-Note
├── phase4_rag_agentic/    # You — FAISS retrieval + LLM-assisted inference
├── mlops_platform/        # You — Docker, CI, MLflow, FastAPI serving, GCP IaC
├── shared/                # Everyone — schemas, config, eval harness
├── data/                  # raw/ processed/ external — gitignored, DVC-tracked
├── docs/                  # architecture diagrams + ADRs
└── scripts/               # one-off setup / data-download scripts
```
