# ADR 0002: FAISS over pgvector/managed vector DB for Phase 4

**Status:** Accepted
**Date:** 2026-07-02

## Context

Phase 4 needs a retrieval index over historical claims (top-k similar past
denied claims, injected as context at inference time). Options considered:
FAISS (local, in-process), pgvector (Postgres extension, and we already run
Postgres for MLflow), a managed vector DB (Pinecone/Weaviate/GCP Vertex
Matching Engine).

## Decision

FAISS (`IndexFlatIP`, upgradeable to `IndexIVFFlat` if the corpus grows past
~100K vectors and exact search gets slow).

## Reasoning

- **Cost:** a managed vector DB adds a billable, always-on service before
  we've validated the retrieval approach even works, and there's no cloud
  deployment in scope for this project. FAISS costs nothing and runs
  in-process.
- **Scale fit:** Synthea-scale synthetic data at ~19% denial prevalence puts
  us in the thousands-to-low-millions of vectors — well within FAISS's
  comfortable range without needing IVF/PQ compression tricks yet.
- **Why not pgvector, given we already run Postgres for MLflow:** coupling
  the retrieval index to the same Postgres instance as experiment tracking
  metadata creates an operational dependency we don't need — if MLflow's DB
  needs a schema migration or restore, we don't want that touching the
  retrieval index. Keeping them separate also makes it trivial to swap FAISS
  for a managed service later without touching MLflow at all.

## Consequences

- Positive: zero marginal infra cost during development; index rebuilds are
  a local, fast operation (`ClaimRetriever.save/load`).
- Negative: FAISS doesn't handle horizontal scaling or high-availability
  serving out of the box — out of scope here, but would need revisiting if
  this ever served production traffic at real hospital scale.
