# M1.8a — Non-destructive molecular identity QA

## Goal

Inspect the chemical structures supplied by BindingDB before deciding which observations correspond to the same molecule. **This stage does not create curated molecule identities or training labels.** The source-reported InChIKey may use different stereochemical or normalization conventions from the supplied SMILES.

## Inputs

`data/interim/M1_7_observation_decisions.csv` — 35,210 observation-level records with construct-policy decisions.

## Method

- Parse each distinct raw SMILES once with RDKit and preserve its original input.
- Compute isomeric canonical SMILES and an RDKit InChIKey, without salt stripping, uncharging, tautomer canonicalization, or stereochemistry removal.
- Compare the source-reported and computed InChIKeys using whole-key equality and first-block equality separately.
- Flag unparseable SMILES, multi-fragment records, and keys with different first blocks for further review.
- Preserve **every** M1.7 observation and its dataset role; do not aggregate IC50 measurements.

## Run from repository root (PowerShell)

```powershell
uv run --locked pytest -q ml/tests/test_audit_molecular_identity.py
uv run --locked ruff check scripts/audit_molecular_identity.py ml/tests/test_audit_molecular_identity.py
uv run --locked python .\scripts\audit_molecular_identity.py `
    --input .\data\interim\M1_7_observation_decisions.csv `
    --output .\data\interim\M1_8a_identity_QA.csv `
    --summary .\reports\audits\M1_8a_identity_QA_summary.json
```

## Interpreting mismatches

A different full InChIKey but matching first block often means that the structures agree in the connectivity layer while differing in other identity layers such as stereochemistry; do **not** equate the two identities automatically. Different first blocks warrant explicit review; they do not automatically establish an error, because standardization or representation may differ.

## Provenance and leakage

The report includes the SHA-256 of the input M1.7 file and RDKit version. Keep row-level output `data/interim/M1_8a_identity_QA.csv` out of the public repository. Do not use source-reported InChIKeys or even calculated SMILES as definitive grouping keys until M1.8b standardization rules and conflict handling have been reviewed.

## Follow-on work (M1.8b)

- Resolve source-vs-computed key disagreements and parse failures.
- Define explicit salt/protonation/tautomer/stereochemical policies, retaining raw evidence.
- Assess overlapping exact-IC50 measurements within target + construct + study context.
- Build curated molecule–target labels and scaffold groups **only after** scientific decisions are finalized.
