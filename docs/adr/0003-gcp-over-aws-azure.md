# ADR 0003: GCP over AWS/Azure for production deployment

**Status:** Accepted
**Date:** 2026-07-02

## Context

We need a cloud target for anything beyond local Docker Compose (public demo
URL, managed Postgres for MLflow if local Postgres isn't durable enough,
potentially GPU access for ClinicalBERT encoding at scale in Phase 3). All
three major providers were considered given free-credit availability for a
student project with no VC/startup-program access.

## Decision

GCP, using the $300/90-day trial credit.

## Reasoning

- **Safety of the free tier matters more than credit size here.** GCP's
  trial has a hard no-auto-charge policy — resources stop when credits run
  out rather than billing a card. AWS's newer credit-based free plan and
  Azure's $200/30-day trial both auto-charge past the limit. For a student
  project with no dedicated cloud budget owner, "fails safe" beats "fails
  expensive."
- **AI/ML-first investment.** GCP's Vertex AI is the most natural fit for
  the ClinicalBERT batch-encoding workload in Phase 3 and for eventually
  hosting the Phase 4 serving layer, without needing to hand-roll
  infrastructure AWS/Azure would require more manual assembly for.
- **Faster credit access.** Azure's largest credits (Founders Hub) target
  startups with a registered business; GCP's trial is immediately available
  to an individual developer, which matches our actual situation (a class
  project, not a company).

## Consequences

- Positive: local dev stays free indefinitely (Docker Compose); GCP is only
  touched for the pieces that need real infra, minimizing credit burn during
  iteration.
- Negative: locks Terraform/IaC work (`mlops_platform/terraform_gcp/`) to GCP-
  specific resources; migrating later would mean rewriting that layer. Judged
  acceptable since Docker Compose (the portable layer) is what teammates
  actually run day to day — GCP is deploy-only, not dev-time infra.
