# M1.4 — Independent BindingDB observation audit: evidence and QA

**Status:** Source-level observation extraction completed; molecule curation **not** completed.
**Run timestamp (UTC):** 2026-10-31T21:09:57.536327+00:00
**Source:** `BindingDB_All_202610_tsv.zip` / `BindingDB_All.tsv`
**Archive SHA-256 (separately verified):** `E968F90E900C7833D5E6ECEAE282D0CA3B75201448BCCAF4DF6B09CC410F7D76`
**Header SHA-256 (reported by audit):** `c8257ee5d10da9a315ac42711845e9397fe92e3b42eee1651af526314054f863`
**Raw archive location:** Outside the repository; no large raw archive committed.

## Scope and rules

- Streamed **3,243,660** TSV records from the uncompressed BindingDB member within the ZIP.
- Matched chain 1's SwissProt primary accession to EGFR `P00533` or HER2 `P04626`; selected only records with exactly one protein chain and nonempty IC50.
- Retained all selected individual experimental observations and publication IDs; did **not** canonicalize structures, aggregate assays, deduplicate molecules, or perform paired-molecule grouping.
- Exact IC50 values become numerical pIC50; inequalities keep censoring direction and lack an exact pIC50. Strict provisional activity cutoff: IC50 **< 1000 nM**.

## Original audit reconciliation

| Population | Count |
| --- | ---: |
| Data records scanned | 3,243,660 |
| Chain-1 EGFR or HER2 accession matches | 38,570 |
| Excluded: multichain or non-`1` chain count | 795 |
| Excluded: missing IC50 (after single-chain filter) | 2,565 |
| Selected observation rows | 35,210 |

**Arithmetic check:** 38,570 − 795 − 2,565 = 35,210.

| Measurement | EGFR | HER2 | Combined |
| --- | ---: | ---: | ---: |
| Exact IC50 | 23,483 | 5,174 | 28,657 |
| Censored IC50 | 5,731 | 821 | 6,552 |
| Invalid IC50 | 1 | 0 | 1 |
| **Total** | **29,215** | **5,995** | **35,210** |

The uploaded observations CSV was independently recounted and agrees with the audit JSON for these fields.

## QA issue 1 — Human organism synonym misclassified

The first-pass script classified only `Homo sapiens` as human and flagged `Human` as `nonhuman_or_unexpected`. The actual CSV contains only three distinct organism strings:

| Raw organism field | Observations | Interpretation |
| --- | ---: | --- |
| `Homo sapiens` | 30,547 | Human |
| `Human` | 3,409 | Human; should not be flagged as nonhuman |
| Blank | 1,254 | Missing metadata; keep flagged |

Therefore **33,956 observations have an explicit human organism label**, and 1,254 have missing organism metadata. The `nonhuman_or_unexpected` count of 3,409 in the first-pass JSON is a metadata-normalization bug, **not** evidence of nonhuman measurements.

**Correction:** recognize case-insensitive `Human` and `Homo sapiens` as human; add a regression test. A corrected audit run should leave scientific observation totals unchanged while updating organism flags and per-target organism counts.

## QA issue 2 — Target construct heterogeneity

The extracted CSV includes numerous target names with bracketed mutations, insertions, deletions or construct ranges. Matching UniProt alone does **not** guarantee identical biochemical targets.

| Source name group | EGFR observations | HER2 observations |
| --- | ---: | ---: |
| Unannotated base target name | 20,997 | 4,736 |
| Annotated or otherwise different target name | 8,218 | 1,259 |

The unannotated name is **not proof** that the assay used a wild-type or full-length protein. M1 must retain a construct field, inspect sequence/variant metadata, and adopt a documented inclusion policy (e.g., baseline unannotated/wild-type-like target vs variant-specific auxiliary analysis) **before** pIC50 aggregation.

Examples in these rows include EGFR `[T790M,L858R]` and HER2 `[V777L]`.

## QA issue 3 — Invalid numeric observation

One EGFR row has raw IC50 `0.000` (source row number `3020894`, `BindingDB Reactant_set_id` `51436180`). It is correctly preserved with `measurement_status=invalid`, no pIC50, and a parse error. **Do not convert zero nM to infinite pIC50.**

## Ruff / code quality

First-pass user checks: **29 pytest tests passed**, but Ruff emitted `SIM117` (nested `with`) and `UP017` (use `datetime.UTC`). The patched script in the companion package folds resource managers into a single parenthesized `with`, uses `UTC`, normalizes the human synonym, and extends the fixture test. A local synthetic-test run of the patch passed; the patched script has not yet been run on the full source archive in the user environment.

## Reproducibility and next gate

1. Apply and test the updated audit source without modifying the raw BindingDB archive.
2. Rerun the full audit into the same generated evidence paths after the code fix, and verify that only organism flags/counts change.
3. Create a target-construct policy before canonical molecule identity and assay reconciliation.
4. Independently compute molecule/pair/scaffold counts; do **not** represent 35,210 observations as 35,210 independent molecules.

**No RF, GNN, generator or drug-effect model has been trained or validated by this audit.**
