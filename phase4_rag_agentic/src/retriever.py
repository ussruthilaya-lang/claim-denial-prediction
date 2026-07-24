"""
Builds and queries a similarity index over historical claims for
retrieval-augmented denial prediction.

WHY RETRIEVAL-AUGMENTED, NOT JUST "BIGGER MODEL":
Phase 2's gradient-boosted model outputs a probability with no explanation a
human can act on. A denial-prevention team doesn't just want "73% chance of
denial" — they want "here are 5 past claims that looked like this one and got
denied, and here's why." That's a fundamentally different product: it's
decision support, not just classification. It is also what powers two concrete
modeling gains this phase measures: (1) retrieval *features* (how did claims
like this one resolve historically?) added to the classifier, and (2) the
agentic LLM in `agentic_predictor.py`, which reasons over retrieved evidence
rather than pattern-matching on structured columns alone.

WHY FAISS (with a dependency-free fallback):
- FAISS's IndexFlatIP is the production backend — it runs entirely in-process,
  zero external service, and at Synthea-scale (thousands to low millions of
  vectors) an exact inner-product index is more than sufficient. See
  docs/adr/0002-faiss-over-pgvector.md.
- FAISS wheels are not always available on brand-new Python builds. Because the
  index here is exact cosine over a small vector count, we provide a NumPy
  backend with byte-identical query semantics so the whole pipeline, tests, and
  demo run on any machine with numpy alone. `ClaimRetriever.backend` reports
  which one is active. The retrieval MATH — normalize, inner product, top-k — is
  identical either way, so results and metrics do not depend on the backend.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import faiss
except ImportError:  # no faiss wheel (e.g. very new Python) -> NumPy fallback
    faiss = None

from shared.schemas.claim import ClaimRecord


@dataclass
class RetrievedClaim:
    claim: ClaimRecord
    similarity: float


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalize; guards zero vectors so cosine == inner product."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (x / norms).astype("float32")


class _NumpyBackend:
    """Exact cosine top-k over a stacked matrix of normalized vectors.
    Same search contract as faiss.IndexFlatIP: returns (similarities, indices)
    with -1 padding when fewer than k vectors exist."""

    def __init__(self, dim: int):
        self.dim = dim
        self._mat = np.empty((0, dim), dtype="float32")

    def add(self, normed: np.ndarray) -> None:
        self._mat = np.vstack([self._mat, normed]) if self._mat.size else normed

    def search(self, q: np.ndarray, k: int):
        if self._mat.shape[0] == 0:
            return np.full((q.shape[0], k), -np.inf), np.full((q.shape[0], k), -1)
        # Inputs are unit-normalized (|sims| <= 1), so there is no real overflow or
        # divide-by-zero here. NumPy still emits spurious "divide by zero / overflow
        # / invalid value encountered in matmul" RuntimeWarnings on some float32
        # BLAS/SIMD paths for perfectly finite inputs — silence those false
        # positives; the computed values are unchanged.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sims = q @ self._mat.T                   # (n_query, n_index)
        k = min(k, sims.shape[1])
        # argpartition for top-k then sort those k descending — O(n) not O(n log n).
        idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(sims, idx, axis=1), axis=1)
        idx = np.take_along_axis(idx, order, axis=1)
        top_sims = np.take_along_axis(sims, idx, axis=1)
        return top_sims, idx

    @property
    def ntotal(self) -> int:
        return self._mat.shape[0]


class ClaimRetriever:
    """
    Wraps a similarity index + a parallel list of ClaimRecords.

    Design note: we keep the index (vectors) and the metadata (ClaimRecord
    objects) as two parallel structures rather than storing metadata inside the
    index itself — the index is a pure vector store, and conflating it with
    metadata storage is a common source of index/metadata drift bugs.
    """

    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        if faiss is not None:
            self.index = faiss.IndexFlatIP(embedding_dim)
            self.backend = "faiss"
        else:
            self.index = _NumpyBackend(embedding_dim)
            self.backend = "numpy"
        self._claims: list[ClaimRecord] = []

    def add(self, embeddings: np.ndarray, claims: list[ClaimRecord]) -> None:
        assert embeddings.shape[0] == len(claims)
        assert embeddings.shape[1] == self.embedding_dim
        self.index.add(_l2_normalize(embeddings))
        self._claims.extend(claims)

    def query(self, embedding: np.ndarray, k: int = 5,
              exclude_ids: set[str] | None = None) -> list[RetrievedClaim]:
        """Top-k most similar historical claims.

        `exclude_ids` drops specific claim_ids from the results — used to prevent
        a claim retrieving itself (the single most important leakage guard in a
        retrieval-augmented setup: without it, a claim's own outcome leaks into
        its own features and the ablation is meaningless)."""
        normed = _l2_normalize(embedding.reshape(1, -1))
        # Over-fetch so post-filtering by exclude_ids can still return k results.
        fetch = k + (len(exclude_ids) if exclude_ids else 0) + 1
        similarities, indices = self.index.search(normed, fetch)
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx == -1:
                continue
            claim = self._claims[idx]
            if exclude_ids and claim.claim_id in exclude_ids:
                continue
            results.append(RetrievedClaim(claim=claim, similarity=float(sim)))
            if len(results) == k:
                break
        return results

    def __len__(self) -> int:
        return len(self._claims)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.backend == "faiss":
            faiss.write_index(self.index, str(path))
        else:
            np.save(path.with_suffix(".vecs.npy"), self.index._mat)
        meta_path = path.with_suffix(".meta.pkl")
        with open(meta_path, "wb") as f:
            pickle.dump({"dim": self.embedding_dim, "backend": self.backend,
                         "claims": [c.model_dump() for c in self._claims]}, f)

    @classmethod
    def load(cls, path: str | Path) -> "ClaimRetriever":
        path = Path(path)
        meta_path = path.with_suffix(".meta.pkl")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        retriever = cls(embedding_dim=meta["dim"])
        if retriever.backend == "faiss":
            retriever.index = faiss.read_index(str(path))
        else:
            retriever.index._mat = np.load(path.with_suffix(".vecs.npy"))
        retriever._claims = [ClaimRecord(**c) for c in meta["claims"]]
        return retriever
