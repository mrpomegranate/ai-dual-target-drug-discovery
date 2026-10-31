# M1.1 — BindingDB archive verification and schema inspection

**Project:** AI Dual-Target Drug Discovery (EGFR/HER2)
**Date:** 2026-10-31
**Status:** Completed — source archive identified and actual TSV header inspected.
**Scope:** Provenance and schema only; no assay rows curated at this step.

## Objective

Establish a concrete, reproducible input source for the capstone. Avoid guessing column names, downloading a second copy, or extracting the full TSV merely to inspect its header.

## Confirmed archive

| Field | Observed value |
|---|---|
| Release file | `BindingDB_All_202610_tsv.zip` |
| ZIP size | `594,284,548` bytes |
| SHA-256 of ZIP | `E968F90E900C7833D5E6ECEAE282D0CA3B75201448BCCAF4DF6B09CC410F7D76` |
| Member | `BindingDB_All.tsv` |
| Member size (uncompressed) | `9,004,582,822` bytes |
| TSV column count | **640** |
| Header SHA-256 | `c8257ee5d10da9a315ac42711845e9397fe92e3b42eee1651af526314054f863` |
| Duplicate or empty header names | None detected |

The ZIP checksum was separately verified on the user's machine; the streaming audit does not recalculate the full ZIP SHA-256.

## Important source fields

The parser uses the real source header, including:

- `BindingDB Reactant_set_id`, `BindingDB MonomerID`, `Ligand SMILES`, `Ligand InChI Key`.
- `Target Name`, `Target Source Organism According to Curator or DataSource`.
- `UniProt (SwissProt) Primary ID of Target Chain 1` and `Number of Protein Chains in Target (>1 implies a multichain complex)`.
- `IC50 (nM)`, `Curation/DataSource`, `Article DOI`, `PMID`, `Patent Number`, `pH`, `Temp (C)`.

Target accession mapping for later stages:

| Target | UniProt Swiss-Prot primary accession |
|---|---|
| EGFR | `P00533` |
| HER2/ERBB2 | `P04626` |

## Reproduce (PowerShell)

Set `$archivePath` to the existing ZIP on your machine (example uses the original data location):

```powershell
$archivePath = 'E:\path\to\BindingDB_All_202610_tsv.zip'

Get-FileHash -Path "$archivePath" -Algorithm SHA256

uv run --locked python .\scripts\inspect_bindingdb_schema.py `
    --input "$archivePath" `
    --out .\reports\audits\bindingdb_202610_schema.json
```

## Saved artifact

`../reports/audits/bindingdb_202610_schema.json` — complete ordered 640-column inventory, size/CRC/header hash and candidate scientific fields.

## Interpretation and limitations

This milestone confirms that the archive and header have the expected structure. A matching header hash does not, by itself, establish biological data quality, assay homogeneity, target specificity, or molecule counts. The target accession and one-chain policies used later are **explicit restrictions**, not an exhaustive account of all possible EGFR/HER2 assays.
