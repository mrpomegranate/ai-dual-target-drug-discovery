# M1.2 — Real EGFR/HER2 record preview

**Project:** AI Dual-Target Drug Discovery (EGFR/HER2)
**Date:** 2026-10-31
**Status:** Completed — genuine observations read directly from October 2026 BindingDB ZIP.
**Scope:** Small schema/field-content inspection; **not** whole-dataset statistics.

## Objective

Verify that actual records can be streamed from the archive and that our target IDs, IC50 column, SMILES and publication provenance fields are populated as expected before a multi-million-row scan.

## Selection rule

Preview rows where chain 1 has UniProt `P00533` (EGFR) or `P04626` (HER2), the record reports **exactly one** protein chain, and IC50 is nonempty. The preview takes three examples per target, stopping after both targets have enough examples.

## Observed output

The script scanned **6,793** input records before collecting six examples:

| BindingDB reactant set ID | Target | IC50 (nM) | Approx. pIC50 |
|---|---|---:|---:|
| `3775` | EGFR | 800 | 6.097 |
| `3776` | EGFR | 220 | 6.658 |
| `3777` | EGFR | 160 | 6.796 |
| `6791` | HER2 | 378 | 6.423 |
| `6792` | HER2 | 114 | 6.943 |
| `6793` | HER2 | 14 | 7.854 |

All six examples had nonempty SMILES, human organism metadata, one chain, and the expected source accessions. The measured values are exact numerals rather than inequalities.

The potency transform used here for exact values is:

```text
pIC50 = 9 - log10(IC50_in_nM)
```

## Reproduce (PowerShell)

Assuming `$archivePath` was set as in M1.1:

```powershell
uv run --locked python .\scripts\preview_bindingdb_targets.py `
    --input "$archivePath" `
    --out .\data\interim\bindingdb_target_preview.json `
    --per-target 3
```

## Saved artifact

`../data/interim/bindingdb_target_preview.json` contains `rows_scanned`, preview-selection metadata and `samples.EGFR` / `samples.HER2` with source IDs, raw IC50, source, DOI/PMID, SMILES and target metadata. The interim JSON may be ignored by Git; retain it locally as audit evidence.

## Scientific caution

The six sample records represent **different molecules**. An EGFR inhibitor and a HER2 inhibitor are not automatically a dual-target inhibitor. Only molecules with comparable experimental evidence for **both** targets can later support a paired dual-target label. This was a smoke test of record semantics, not an estimate of prevalence, source bias, or assay reliability.
