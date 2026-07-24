"""
Semantic evidence retriever using sentence-transformers (all-MiniLM-L6-v2).
Embeddings are computed once per unique text (cached) and reused across cases —
this is the 'main method' compared against the TF-IDF baseline.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL = None
_CACHE = {}

def load_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL

def _embed(text):
    if text not in _CACHE:
        _CACHE[text] = load_model().encode(text, normalize_embeddings=True)
    return _CACHE[text]

def best_match(query_text, candidate_chunks):
    if not candidate_chunks:
        return None, 0.0
    q = _embed(query_text)
    sims = [float(np.dot(q, _embed(ch))) for ch in candidate_chunks]
    best_idx = int(np.argmax(sims))
    return candidate_chunks[best_idx], sims[best_idx]

def classify_case(case, threshold=0.35):
    sub_chunk, sub_score = best_match(case["requirement_text"], case["submitted_chunks"])
    rec_chunk, rec_score = best_match(case["requirement_text"], case["full_record_chunks"])

    if sub_score >= threshold:
        status, cited = "complete", sub_chunk
    elif rec_score >= threshold:
        status, cited = "omitted", rec_chunk
    else:
        status, cited = "unsupported", None

    return {
        "case_id": case["case_id"], "predicted_status": status, "gold_variant": case["gold_variant"],
        "sub_score": round(sub_score, 3), "rec_score": round(rec_score, 3), "cited_chunk": cited,
    }