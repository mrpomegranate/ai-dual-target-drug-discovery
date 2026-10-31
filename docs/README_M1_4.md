# M1.4 — Independent whole-archive observation extraction

**Project:** AI Dual-Target Drug Discovery (EGFR/HER2)
**Date:** 2026-10-31
**Status:** Completed — full ZIP streamed, counts reconciled, organism flag corrected, code tests/lint passed.
**Scope:** **Observation-level** evidence extraction; not final model-ready compound labels.

## Objective and inclusion policy

Read `BindingDB_All.tsv` directly from the October 2026 ZIP, preserve source-level experimental observations for EGFR and HER2, and explicitly record exclusions and measurement parsing status. To make this first pass well defined, only rows with chain 1 Swiss-Prot accession `P00533` (EGFR) / `P04626` (HER2) and `Number of Protein Chains... == 1` are extracted. Records missing IC50 are excluded from the export **but counted**. Invalid IC50 records remain flagged. No duplicate merging or canonicalization is performed.

## Independently observed extraction counts

| Quantity | EGFR | HER2 | Combined |
|---|---:|---:|---:|
| Accession-matched records | 31,585 | 6,985 | **38,570** |
| Excluded multichain | 15 | 780 | **795** |
| Excluded missing IC50 | 2,355 | 210 | **2,565** |
| Observations written | **29,215** | **5,995** | **35,210** |
| Exact IC50 | 23,483 | 5,174 | **28,657** |
| Censored IC50 | 5,731 | 821 | **6,552** |
| Invalid IC50 | 1 | 0 | **1** |
| Explicitly human metadata | 28,371 | 5,585 | **33,956** |
| Missing organism metadata | 844 | 410 | **1,254** |

**Entire archive scanned:** `3,243,660` source records.
**Reconciliation:** `38,570 − 795 − 2,565 = 35,210` and `28,657 + 6,552 + 1 = 35,210`.

The invalid measurement was raw IC50 `0.000` on input record number `3,020,894`. It was **retained as invalid** and never assigned a pIC50 value.

## QA correction made during M1.4

An early organism classifier counted `Human` as unexpected while recognizing `Homo sapiens`. The corrected implementation recognizes **both strings** as explicitly human, without filling missing organism fields. Following the fix:

- Human metadata: **33,956**.
- Missing organism metadata: **1,254**.
- Unexpected organism values: **0** in the corrected summary (the key may be omitted because its count is zero).
- Total observations and IC50-type counts were unchanged.

The script also received Ruff fixes for `SIM117` (context managers) and `UP017` (`datetime.UTC`). After applying them, the full local project suite was **29 tests passed**, with Ruff passing; this was the test count at the conclusion of M1.4.

## Outputs

| Path | Role |
|---|---|
| `../scripts/audit_bindingdb_observations.py` | Streaming, reproducible observation audit |
| `../data/interim/bindingdb_egfr_her2_observations.csv` | Source-level observation table, typically kept outside Git |
| `../reports/audits/bindingdb_egfr_her2_observation_audit.json` | Machine-readable counts, policies, limitations and source metadata |
| `../reports/milestones/M1_4_observation_evidence.md` | Additional QA evidence if copied from the correction bundle |

The CSV retains original records, reported SMILES/InChIKeys, source/DOI/PMID/patent, target name, IC50 qualifier, pIC50 only for exact values, and data-quality flags.

## Reproduce (PowerShell)

Assuming `$archivePath` points to the original ZIP:

```powershell
uv run --locked pytest -q
uv run --locked ruff check scripts/audit_bindingdb_observations.py ml/src ml/tests

uv run --locked python .\scripts\audit_bindingdb_observations.py `
    --input "$archivePath" `
    --observations .\data\interim\bindingdb_egfr_her2_observations.csv `
    --summary .\reports\audits\bindingdb_egfr_her2_observation_audit.json
```

Optional inspection:

```powershell
Get-Content .\reports\audits\bindingdb_egfr_her2_observation_audit.json -Raw
```

## Limitations and decision gate

- These are **assay observations, not unique compounds**, paired measurements or validated drug-target interactions.
- The match excludes multichain targets and only considers chain 1's primary accession.
- `human` target metadata and matching accession are not sufficient to prove that a record used the same protein **construct** as other records; M1.6 examines annotations.
- Target annotations, assay protocols, censoring, source duplication, molecule stereochemistry/chemistry and missing target labels must be handled before regression or classification training.
- UTF-8 decoding uses replacement for malformed bytes; this is disclosed as an extraction limitation.
