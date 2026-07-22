"""
Feature engineering for Phase 4.

Two distinct feature sets live here, and keeping them separate is the whole
methodological point of the phase:

1. STRUCTURED FEATURES (`StructuredEncoder`) — the observable billing fields a
   Phase-1/2 model would use: payer, procedure, diagnosis category, log billed
   amount, amount-vs-fee-schedule ratio, and filing delay. These double as the
   retrieval EMBEDDING: two claims are "similar" when their structured profiles
   are close. Nothing here touches the label or any latent driver, so the
   embedding a claim retrieves with is exactly what a biller could see.

2. RETRIEVAL FEATURES (`RetrievalFeaturizer`) — computed by asking the index
   "how did the k most similar HISTORICAL claims resolve?" (neighbour denial
   rate, similarity-weighted denial score, agreement, amount deltas). This is
   the retrieval-augmented signal Phase 4 adds on top of the structured model.

   The leakage guard is essential and explicit: the index is built only over
   TRAINING claims, and a training claim excludes ITSELF from its own neighbours
   (see `exclude_self`). Skip either and a claim's own outcome leaks into its
   own features, inflating the ablation into nonsense — exactly the trap a
   retrieval-augmented pipeline invites.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from phase4_rag_agentic.src.data_gen import ICD10, PROCEDURE_BY_CPT
from phase4_rag_agentic.src.retriever import ClaimRetriever, _l2_normalize

# Reverse map ICD-10 code -> clinical category (public grouping, observable).
_ICD_TO_CAT = {code: cat for cat, codes in ICD10.items() for code in codes}


def _filing_days(frame: pd.DataFrame) -> np.ndarray:
    return (pd.to_datetime(frame["submission_date"]) -
            pd.to_datetime(frame["service_date"])).dt.days.to_numpy(dtype=float)


class StructuredEncoder:
    """Fit-on-train, transform-anywhere encoder for the observable claim fields.

    Categorical vocabularies and numeric standardization stats are learned once
    on the training frame, so the index, the training matrix, and any query at
    inference all land in the same vector space (a frequent silent bug when each
    split one-hot-encodes independently)."""

    # Weight on the provider block when building the RETRIEVAL embedding. The
    # classifier never sees provider_id (too high-cardinality to one-hot into a
    # tree usefully), so the retrieval layer is the only thing that can organize
    # history by provider and surface provider-level denial propensity — the
    # latent "clean-claim discipline" that actually drives many denials. This is
    # what makes retrieval features add signal the flat model cannot reach.
    PROVIDER_WEIGHT = 5.0

    def __init__(self):
        self.payers: list[str] = []
        self.cpts: list[str] = []
        self.cats: list[str] = []
        self.providers: list[str] = []
        self.num_mean = None
        self.num_std = None
        self.feature_names: list[str] = []

    def _numeric(self, frame: pd.DataFrame) -> np.ndarray:
        base_cost = frame["cpt_code"].map(
            lambda c: PROCEDURE_BY_CPT[c].base_cost).to_numpy(dtype=float)
        billed = frame["billed_amount"].to_numpy(dtype=float)
        return np.column_stack([
            np.log1p(billed),                     # scale of the claim
            billed / base_cost,                   # amount vs fee schedule (upcoding)
            _filing_days(frame),                  # timely-filing exposure
        ])

    def fit(self, frame: pd.DataFrame) -> "StructuredEncoder":
        self.payers = sorted(frame["insurance_type"].unique())
        self.cpts = sorted(frame["cpt_code"].unique())
        self.cats = sorted({_ICD_TO_CAT[c] for c in frame["icd10_code"].unique()})
        self.providers = sorted(frame["provider_id"].unique())
        num = self._numeric(frame)
        self.num_mean = num.mean(axis=0)
        self.num_std = num.std(axis=0) + 1e-8
        self.feature_names = (
            ["log_billed", "amount_ratio", "filing_days"]
            + [f"payer={p}" for p in self.payers]
            + [f"cpt={c}" for c in self.cpts]
            + [f"dx={c}" for c in self.cats]
        )
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        num = (self._numeric(frame) - self.num_mean) / self.num_std
        payer = np.array([[1.0 if p == v else 0.0 for v in self.payers]
                          for p in frame["insurance_type"]])
        cpt = np.array([[1.0 if c == v else 0.0 for v in self.cpts]
                        for c in frame["cpt_code"]])
        cat = np.array([[1.0 if _ICD_TO_CAT[c] == v else 0.0 for v in self.cats]
                        for c in frame["icd10_code"]])
        return np.hstack([num, payer, cpt, cat]).astype("float32")

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)

    def transform_embedding(self, frame: pd.DataFrame) -> np.ndarray:
        """Retrieval EMBEDDING = classifier features + a weighted provider block.

        Two claims are 'similar' when they share billing profile AND provider.
        Provider is deliberately excluded from `transform` (the classifier's
        features) so any lift the augmented model shows is signal the retrieval
        layer contributed, not something the flat model already had."""
        base = self.transform(frame)
        prov = np.array([[1.0 if p == v else 0.0 for v in self.providers]
                         for p in frame["provider_id"]], dtype="float32")
        return np.hstack([base, self.PROVIDER_WEIGHT * prov]).astype("float32")


RETRIEVAL_FEATURE_NAMES = [
    "retr_denial_rate",        # unweighted fraction of neighbours denied
    "retr_weighted_denial",    # similarity-weighted fraction denied
    "retr_mean_similarity",    # how close the neighbourhood actually is
    "retr_max_similarity",     # nearest-neighbour closeness
    "retr_n_denied",           # count of denied neighbours
    "retr_billed_delta",       # own billed / mean neighbour billed
    "retr_sim_spread",         # top1 - mean similarity (is the match sharp?)
]


class RetrievalFeaturizer:
    """Turns k retrieved neighbours into a fixed vector of retrieval features.

    The neighbour labels/amounts/ids of the indexed claims are cached once as
    NumPy arrays so featurizing is a single batched similarity search plus cheap
    per-row aggregation over tiny (k+buffer,) slices — seconds at 40k claims,
    instead of a per-row Python search loop."""

    def __init__(self, retriever: ClaimRetriever, k: int = 10):
        self.retriever = retriever
        self.k = k
        self._denied = np.array(
            [1.0 if c.denied else 0.0 for c in retriever._claims], dtype="float32")
        self._billed = np.array(
            [c.billed_amount for c in retriever._claims], dtype="float32")
        self._ids = np.array([c.claim_id for c in retriever._claims])

    def transform(self, embeddings: np.ndarray, frame: pd.DataFrame,
                  exclude_self: bool) -> np.ndarray:
        """Compute retrieval features for every row.

        exclude_self=True for the TRAINING frame (a claim is in the index and
        must not retrieve itself); False for held-out/test claims and live
        inference (they are not in the index at all)."""
        own_ids = frame["claim_id"].to_numpy()
        own_billed = frame["billed_amount"].to_numpy(dtype=float)
        # Over-fetch by one so we can drop a self-match and still keep k.
        fetch = self.k + 1
        normed = _l2_normalize(np.asarray(embeddings, dtype="float32"))
        sims_all, idx_all = self.retriever.index.search(normed, fetch)

        out = np.empty((len(frame), len(RETRIEVAL_FEATURE_NAMES)), dtype="float32")
        for i in range(len(frame)):
            idx = idx_all[i]
            sims = sims_all[i]
            valid = idx != -1
            idx, sims = idx[valid], sims[valid]
            if exclude_self:
                keep = self._ids[idx] != own_ids[i]
                idx, sims = idx[keep], sims[keep]
            idx, sims = idx[:self.k], np.clip(sims[:self.k], 0.0, None)
            if idx.size == 0:
                out[i] = 0.0
                continue
            denied = self._denied[idx]
            billed = self._billed[idx]
            w = sims / (sims.sum() + 1e-8)
            out[i] = (
                denied.mean(),
                float((w * denied).sum()),
                float(sims.mean()),
                float(sims.max()),
                float(denied.sum()),
                float(own_billed[i] / (billed.mean() + 1e-8)),
                float(sims.max() - sims.mean()),
            )
        return out
