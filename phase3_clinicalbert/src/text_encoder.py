"""
Pluggable clinical-note encoder for Phase 3.

The phase's scientific question — "does the clinical narrative carry denial
signal the structured billing fields do not?" — does not depend on *which* text
encoder we use, only on whether the note is embedded per-claim and fused with the
structured features. So we keep the encoder swappable behind one interface:

  * TfidfTextEncoder   — fast, no heavy dependencies, runs anywhere. The default,
                         used for the trial run so the architecture can be
                         validated in seconds instead of the ~74 minutes the
                         prototype's ClinicalBERT pass took.
  * ClinicalBertEncoder — the proposal's Bio_ClinicalBERT CLS embedding. Same
                         interface; used for the final numbers once torch +
                         transformers are installed. Import is lazy so this
                         module loads with neither installed.

Both fit on TRAINING notes only and transform anywhere, mirroring
StructuredEncoder, so nothing leaks from test into the fitted vocabulary/stats.
"""
from __future__ import annotations

import numpy as np


class TfidfTextEncoder:
    """TF-IDF bag-of-words (uni+bigram) over the note text. Cheap and strong on
    templated notes; a reasonable stand-in for a contextual encoder when the
    goal is to measure whether the *content* of the note is predictive."""

    name = "tfidf"

    def __init__(self, max_features: int = 400, ngram_range=(1, 2),
                 min_df: int = 5):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(max_features=max_features,
                                   ngram_range=ngram_range, min_df=min_df,
                                   stop_words="english")
        self.dim = None

    def fit(self, notes: list[str]) -> "TfidfTextEncoder":
        self.vec.fit(notes)
        self.dim = len(self.vec.vocabulary_)
        return self

    def transform(self, notes: list[str]) -> np.ndarray:
        return self.vec.transform(notes).astype("float32").toarray()

    def fit_transform(self, notes: list[str]) -> np.ndarray:
        return self.fit(notes).transform(notes)


class ClinicalBertEncoder:
    """Bio_ClinicalBERT CLS-token embedding (768-d), batched, no grad, optional
    on-disk cache. Requires torch + transformers; imported lazily so the module
    is usable without them. This is the proposal's Phase 3 encoder."""

    name = "clinicalbert"

    def __init__(self, model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
                 max_length: int = 256, batch_size: int = 32,
                 device: str | None = None):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:  # pragma: no cover - depends on optional deps
            raise ImportError(
                "ClinicalBertEncoder needs torch + transformers "
                "(`pip install torch transformers`). Use encoder='tfidf' to run "
                "without them.") from e
        self._torch = torch
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():  # Apple Silicon
            self.device = "mps"
        else:
            self.device = "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.max_length = max_length
        self.batch_size = batch_size
        self.dim = 768

    def fit(self, notes: list[str]) -> "ClinicalBertEncoder":
        return self  # a pretrained encoder needs no fitting

    def transform(self, notes: list[str]) -> np.ndarray:
        torch = self._torch
        out = []
        for i in range(0, len(notes), self.batch_size):
            batch = [str(t) for t in notes[i:i + self.batch_size]]
            enc = self.tokenizer(batch, return_tensors="pt", truncation=True,
                                 max_length=self.max_length, padding=True).to(self.device)
            with torch.no_grad():
                cls = self.model(**enc).last_hidden_state[:, 0, :]
            out.append(cls.cpu().numpy())
        return np.vstack(out).astype("float32")

    def fit_transform(self, notes: list[str]) -> np.ndarray:
        return self.transform(notes)


def get_text_encoder(name: str = "tfidf", **kwargs):
    name = (name or "tfidf").lower()
    if name == "tfidf":
        return TfidfTextEncoder(**kwargs)
    if name in ("clinicalbert", "bert", "bio_clinicalbert"):
        return ClinicalBertEncoder(**kwargs)
    raise ValueError(f"unknown text encoder {name!r} (use 'tfidf' or 'clinicalbert')")
