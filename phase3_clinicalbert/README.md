# Phase 3 — ClinicalBERT (owner: Teammate C)

Encodes MIMIC-IV-Note discharge summaries with ClinicalBERT, concatenates
embeddings with structured features, retrains the Phase 2 classifier. Also
evaluates a GPT-4 zero-shot baseline for comparison.

## Before you start

MIMIC-IV-Note requires free CITI training + PhysioNet credentialing — do not
place any raw discharge summaries anywhere in this repo (see `.gitignore`,
which blocks `phase3_clinicalbert/data/*` entirely; this is intentional and
should not be relaxed). Set `MIMIC_NOTE_DATA_DIR` in `.env` to point at your
local, credentialed copy outside the repo.

## Contract with the rest of the repo

Populate `ClaimRecord.clinical_note_text` only for records with confirmed
credentialed access. Since MIMIC-IV-Note has no native denial labels, use
ICD-10/CPT proxy label construction (per the proposal, Sec. 3) and document
the proxy logic clearly in `src/proxy_labels.py` — this is a judgment call
future readers (and Phase 4's retrieval evidence) need to be able to audit.

## Structure

```
phase3_clinicalbert/
├── data/           # gitignored entirely — credentialed data only, never committed
├── notebooks/
├── src/            # encode.py, proxy_labels.py, fuse_features.py, train.py
└── tests/
```
