"""
Builds and queries a FAISS index over historical claims for retrieval-augmented
denial prediction.

WHY RETRIEVAL-AUGMENTED, NOT JUST "BIGGER MODEL":
Phase 2's XGBoost outputs a probability with no explanation a human can act on.
A denial-prevention team doesn't just want "73% chance of denial" — they want
"here are 5 past claims that looked like this one and got denied, and here's
why." That's a fundamentally different product: it's decision support, not
just classification. This is also the part of the pipeline that's genuinely
"agentic" — the LLM in `agentic_predictor.py` reasons over retrieved evidence
rather than pattern-matching on structured columns alone.

WHY FAISS SPECIFICALLY (not pgvector, not a managed vector DB):
- Runs entirely locally/in-process — zero GCP cost during dev, which matters
  since we're on a 90-day credit clock.
- At ~19% denial prevalence over Synthea-scale synthetic records, this is a
  small-to-mid vector count (thousands-low millions) — FAISS's IndexFlatIP or
  IndexIVFFlat is more than sufficient; we don't need a managed service's
  operational overhead yet. See docs/adr/0002-faiss-over-pgvector.md.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import faiss
except ImportError:  # keep import-time cost low for environments that only test schemas
    faiss = None

from shared.schemas.claim import ClaimRecord


@dataclass
class RetrievedClaim:
    claim: ClaimRecord
    similarity: float


class ClaimRetriever:
    """
    Wraps a FAISS flat inner-product index + a parallel list of ClaimRecords.

    Design note: we keep the index (vectors) and the metadata (ClaimRecord
    objects) as two parallel structures rather than trying to store metadata
    inside FAISS itself — FAISS is a pure vector store, and conflating it with
    metadata storage is a common source of index/metadata drift bugs.
    """

    def __init__(self, embedding_dim: int):
        if faiss is None:
            raise ImportError("faiss-cpu not installed — `pip install faiss-cpu`")
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)  # cosine sim on normalized vecs
        self._claims: list[ClaimRecord] = []

    def add(self, embeddings: np.ndarray, claims: list[ClaimRecord]) -> None:
        assert embeddings.shape[0] == len(claims)
        assert embeddings.shape[1] == self.embedding_dim
        normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.index.add(normed.astype("float32"))
        self._claims.extend(claims)

    def query(self, embedding: np.ndarray, k: int = 5) -> list[RetrievedClaim]:
        normed = embedding / np.linalg.norm(embedding)
        normed = normed.astype("float32").reshape(1, -1)
        similarities, indices = self.index.search(normed, k)
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx == -1:
                continue
            results.append(RetrievedClaim(claim=self._claims[idx], similarity=float(sim)))
        return results

    def save(self, path: str | Path) -> None:
        path = Path(path)
        faiss.write_index(self.index, str(path))
        meta_path = path.with_suffix(".meta.pkl")
        with open(meta_path, "wb") as f:
            pickle.dump([c.model_dump() for c in self._claims], f)

    @classmethod
    def load(cls, path: str | Path) -> "ClaimRetriever":
        path = Path(path)
        index = faiss.read_index(str(path))
        meta_path = path.with_suffix(".meta.pkl")
        with open(meta_path, "rb") as f:
            raw_claims = pickle.load(f)
        retriever = cls.__new__(cls)
        retriever.embedding_dim = index.d
        retriever.index = index
        retriever._claims = [ClaimRecord(**c) for c in raw_claims]
        return retriever
