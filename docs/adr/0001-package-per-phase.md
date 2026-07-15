# ADR 0001: Package-per-phase + shared platform layer

**Status:** Accepted
**Date:** 2026-07-02

## Context

Three people, four phases, one final ablation study comparing all phases'
AUROC/F1. Risk: notebooks that only run on one person's machine, metrics
computed inconsistently across phases, no way to demo a working system at
the end.

## Decision

Each phase is an independent installable package with its own `src/`,
`tests/`, `requirements.txt`. A `shared/` package holds the one thing every
phase must agree on (the `ClaimRecord` schema, the eval harness). A
`mlops_platform/` package owns turning four models' finished results into
one thing to look at: a Streamlit demo app built last, over saved
predictions/SHAP output/retrieval examples from each phase.

**2026-07-15 update:** the original version of this decision also scoped
Docker Compose, MLflow, a FastAPI serving layer, and CI into
`mlops_platform/` from day one. That's cut — see the (removed) GCP ADR.
Building serving infra before a single model exists was solving a problem
we didn't have yet. The platform layer's job in this project is narrower:
present finished results, not serve them.

## Consequences

- Positive: teammates can work in parallel without stepping on each other's
  code.
- Positive: the demo app is still a real payoff for the platform-facing
  teammate — a working comparison view across all four phases is more
  convincing at presentation time than four separate notebooks.
- Negative: more upfront structure than a notebook-first approach; teammates
  need to `pip install -e` their package rather than just running cells.
  Mitigated by keeping `notebooks/` available in each phase for exploration
  before code moves into `src/`.
