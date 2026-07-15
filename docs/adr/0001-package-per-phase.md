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
`mlops_platform/` package owns everything that turns four models into one system:
Docker Compose for local dev, MLflow for tracking, a FastAPI serving layer,
CI.

## Consequences

- Positive: teammates can work in parallel without stepping on each other's
  code; the final demo is "hit an API endpoint," not "run four notebooks in
  sequence and hope."
- Positive: `mlops_platform/` becomes the reusable artifact — the same MLOps
  pattern applies to the next project, which is the actual point of owning
  this layer for your career track (this pattern generalizes to any
  multi-model production system, not just claim denial).
- Negative: more upfront structure than a notebook-first approach; teammates
  need to `pip install -e` their package rather than just running cells.
  Mitigated by keeping `notebooks/` available in each phase for exploration
  before code moves into `src/`.
