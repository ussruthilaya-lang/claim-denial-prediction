"""
Phase 3 — LLM zero-shot baseline (the proposal's "GPT-4 zero-shot", on Claude).

Reads each generated clinical note and asks a Claude model to estimate the
denial probability zero-shot, with NO training on our labels. Reported through
the shared eval so it drops into the cross-phase ablation next to the
structured / +text / +retrieval rows. The question it answers: can a frontier
LLM read the note and judge denial risk without being trained on our data?

Cost / throughput design (thousands of short calls):
  * Batch API      — 50% cheaper, async; a baseline is offline, so ideal.
  * Prompt caching — the instructions + schema prefix is identical every call.
  * Structured out — force {denial_probability, denied} so we get a CONTINUOUS
                     score for AUROC (LLM probabilities aren't calibrated, but
                     AUROC only needs correct ranking).
  * Default model  — claude-haiku-4-5 (plenty for yes/no, cheapest). Pass
                     --model claude-sonnet-5 for a stronger baseline.

Gated on a key: if ANTHROPIC_API_KEY (or settings.anthropic_api_key) is unset,
this no-ops so the rest of the project still runs key-free. Requires the SDK:
`pip install anthropic` (imported lazily, so this module imports without it).

    python phase3_clinicalbert/src/llm_baseline.py                 # 2000-note subsample, Haiku
    python phase3_clinicalbert/src/llm_baseline.py --sample 0       # full test set
    python phase3_clinicalbert/src/llm_baseline.py --model claude-sonnet-5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from phase4_rag_agentic.src.data_gen import generate_dataset  # noqa: E402
from phase4_rag_agentic.src.pipeline import _temporal_split  # noqa: E402
from shared.config.settings import settings  # noqa: E402
from shared.utils.eval import evaluate  # noqa: E402

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
PHASE = "phase3"

SYSTEM_PROMPT = (
    "You are a medical-claims adjudication assistant. You are given the free-text "
    "clinical note for a healthcare encounter. Estimate the probability that the "
    "resulting insurance claim will be DENIED by the payer.\n\n"
    "Weigh whether the note documents medical necessity and justification for the "
    "procedure — e.g. failed conservative therapy, symptom severity, guideline "
    "support. Notes that do NOT document such justification are more likely to be "
    "denied for medical necessity.\n\n"
    "Return ONLY the structured object: denial_probability is your calibrated "
    "estimate in [0, 1]; denied is true when denial_probability >= 0.5."
)

# Numeric range/length constraints aren't enforceable in structured-output schemas,
# so denial_probability is validated (clamped to [0,1]) client-side.
_SCHEMA = {
    "type": "object",
    "properties": {
        "denial_probability": {"type": "number"},
        "denied": {"type": "boolean"},
    },
    "required": ["denial_probability", "denied"],
    "additionalProperties": False,
}


def _resolve_key() -> str | None:
    return getattr(settings, "anthropic_api_key", None) or os.getenv("ANTHROPIC_API_KEY")


def _params(note: str, model: str) -> dict:
    p = {
        "model": model,
        "max_tokens": 200,
        # Cached prefix: instructions are identical across every note (~90% off).
        "system": [{"type": "text", "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"}}],
        "output_config": {"format": {"type": "json_schema", "schema": _SCHEMA}},
        "messages": [{"role": "user", "content": note}],
    }
    # A yes/no classification doesn't need reasoning; disable it where the model
    # would otherwise think by default (Sonnet 5 / Opus). Haiku 4.5 doesn't think
    # unless asked, so leave its thinking field unset.
    if not model.startswith("claude-haiku"):
        p["thinking"] = {"type": "disabled"}
    return p


def _extract_prob(message) -> float:
    for block in message.content:
        if getattr(block, "type", None) == "text":
            try:
                obj = json.loads(block.text)
                return float(np.clip(float(obj["denial_probability"]), 0.0, 1.0))
            except Exception:
                break
    return 0.5  # neutral fallback if the model didn't return parseable JSON


def predict_denial_llm(notes: list[str], model: str = "claude-haiku-4-5",
                       poll_seconds: int = 15, max_wait_minutes: int = 60) -> np.ndarray:
    """Zero-shot denial probability per note via the Batch API. Returns np.ndarray
    of shape (len(notes),) with values in [0,1]; errored/unfinished rows get 0.5."""
    import anthropic  # lazy: module imports fine without the SDK installed

    key = _resolve_key()
    if not key:
        raise RuntimeError(
            "No Anthropic key found. Set ANTHROPIC_API_KEY in your .env to run the "
            "LLM baseline (it no-ops without one).")
    client = anthropic.Anthropic(api_key=key)

    requests = [{"custom_id": f"note-{i}", "params": _params(n, model)}
                for i, n in enumerate(notes)]
    batch = client.messages.batches.create(requests=requests)
    print(f"batch {batch.id} submitted — {len(requests)} notes, model={model}")

    deadline = time.time() + max_wait_minutes * 60
    while True:
        b = client.messages.batches.retrieve(batch.id)
        if b.processing_status == "ended":
            break
        if time.time() > deadline:
            print(f"  batch not finished after {max_wait_minutes} min — "
                  f"collecting whatever succeeded")
            break
        time.sleep(poll_seconds)

    probs = np.full(len(notes), 0.5, dtype=float)
    errors = 0
    for res in client.messages.batches.results(batch.id):
        idx = int(res.custom_id.split("-")[1])
        if res.result.type == "succeeded":
            probs[idx] = _extract_prob(res.result.message)
        else:
            errors += 1
    if errors:
        print(f"  {errors} request(s) errored — assigned neutral 0.5")
    return probs


def run_llm_baseline(n: int = 40_000, seed: int = 42,
                     model: str = "claude-haiku-4-5", sample: int = 2000) -> dict:
    df = generate_dataset(n=n, seed=seed)
    _, test = _temporal_split(df)
    if sample and 0 < sample < len(test):
        test = test.sample(n=sample, random_state=seed).reset_index(drop=True)
    notes = test["clinical_note"].tolist()
    y = test["denied"].to_numpy()

    probs = predict_denial_llm(notes, model=model)
    r = evaluate(y, probs, PHASE, f"llm-zeroshot-{model}")
    return {
        "phase": PHASE,
        "model": model,
        "n_scored": int(len(y)),
        "test_prevalence": float(y.mean()),
        "llm_zeroshot": r.as_dict(),
    }


def main(n: int = 40_000, seed: int = 42, model: str = "claude-haiku-4-5",
         sample: int = 2000, save: bool = True) -> dict:
    if not _resolve_key():
        print("No ANTHROPIC_API_KEY set — skipping the LLM baseline (this is fine; "
              "set the key in .env and re-run to produce the row).")
        return {}
    m = run_llm_baseline(n=n, seed=seed, model=model, sample=sample)
    r = m["llm_zeroshot"]
    print("\n================ PHASE 3 — LLM ZERO-SHOT BASELINE ================")
    print(f"model={m['model']}  n_scored={m['n_scored']}  "
          f"prevalence={m['test_prevalence']:.3f}")
    print(f"AUROC={r['auroc']:.4f}  F1={r['f1']:.4f}  "
          f"P={r['precision']:.4f}  R={r['recall']:.4f}")
    print("Compare against the structured baseline (0.733) and +text (0.769) rows.")
    if save:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        out = ARTIFACT_DIR / f"llm_baseline_{model}.json"  # model-specific: keeps every run
        with open(out, "w") as f:
            json.dump(m, f, indent=2)
        print(f"metrics -> {out}")
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--sample", type=int, default=2000,
                    help="notes to score from the test set; 0 = full test set")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    main(n=args.n, seed=args.seed, model=args.model, sample=args.sample,
         save=not args.no_save)
