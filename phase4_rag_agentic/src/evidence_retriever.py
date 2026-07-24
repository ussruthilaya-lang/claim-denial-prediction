"""
Requirement-level evidence retriever. TF-IDF is fit ONCE globally (stable
vocabulary/IDF weights, like a real embedding index); retrieval is scoped
per-case at lookup time only, so no cross-patient leakage.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

_VECTORIZER = None

def build_global_index(cases):
    global _VECTORIZER
    all_texts = set()
    for c in cases:
        all_texts.add(c["requirement_text"])
        all_texts.update(c["full_record_chunks"])
    _VECTORIZER = TfidfVectorizer(stop_words="english").fit(list(all_texts))
    return _VECTORIZER

def _vec(text):
    return _VECTORIZER.transform([text]).toarray()[0]

def _cos(a, b):
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-9) * (np.linalg.norm(b) + 1e-9)))

def best_match(query_text, candidate_chunks):
    if not candidate_chunks:
        return None, 0.0
    q = _vec(query_text)
    sims = [_cos(q, _vec(ch)) for ch in candidate_chunks]
    best_idx = int(np.argmax(sims))
    return candidate_chunks[best_idx], sims[best_idx]

def classify_case(case, threshold=0.15):
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