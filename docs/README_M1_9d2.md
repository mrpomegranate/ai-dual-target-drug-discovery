# M1.9d.2 — Focused Original-Source Verification Casebook

## Purpose

M1.9d.1 identified 656 publication/patent provenance groups for 20 deliberately extreme molecule–target cases. M1.9d.2 builds a **bounded, manual-assay-verification casebook** rather than attempting to manually adjudicate the entire BindingDB dataset.

The script joins `M1_9d1_source_review_queue.csv` to the original `M1_9c_priority_observation_evidence.csv`, validates **every** source-group row count, IC50 minimum/maximum, pIC50 range, and activity-threshold crossing, then selects a focused and reproducible sample.

**No IC50 is averaged or removed. No model label is created. No protein construct is auto-corrected.**

## Files

| File | Purpose | Commit to Git? |
|---|---|---|
| `scripts/build_original_source_casebook.py` | Reproducible sampling and provenance validation | Yes |
| `ml/tests/test_original_source_casebook.py` | Unit tests, including non-destructive scientific safeguards | Yes |
| `reports/audits/M1_9d2_source_casebook_summary.json` | Aggregate report and input fingerprints | Yes |
| `reports/audits/M1_9d2_verified_publication_leads.md` | Publication-level evidence and unresolved questions | Yes |
| `reports/figures/assay_disagreement/17_original_source_review_selection.png` | Selected group distribution | Yes |
| `data/interim/M1_9d2_source_verification_worksheet.csv` | Manual row-to-source verification worksheet | **No — keep local** |
| `data/interim/M1_9d2_selected_source_observations.csv` | Individual original BindingDB measurements in selected groups | **No — keep local** |

## Validate

From your repository root (Windows PowerShell):

```powershell
uv run --locked pytest -q
uv run --locked ruff check scripts/build_original_source_casebook.py ml/tests/test_original_source_casebook.py
```

## Reproduce the casebook

```powershell
uv run --locked python .\scripts\build_original_source_casebook.py `
    --queue .\data\interim\M1_9d1_source_review_queue.csv `
    --evidence .\data\interim\M1_9c_priority_observation_evidence.csv `
    --worksheet .\data\interim\M1_9d2_source_verification_worksheet.csv `
    --selected-records .\data\interim\M1_9d2_selected_source_observations.csv `
    --summary .\reports\audits\M1_9d2_source_casebook_summary.json `
    --figure .\reports\figures\assay_disagreement\17_original_source_review_selection.png `
    --per-target 6
```

With the audited inputs supplied for this stage, the preview selects **14 source groups** (8 EGFR and 6 HER2) comprising **44 observation records**, including two additional literature leads beyond the 6-per-target core selection. The local rerun is authoritative.

## How to complete the manual review (one case at a time)

For a given `target` + `m18c_structure_id` + `provenance_key` group:

1. Use `lookup_url` to locate the original paper. `publication_context_review=pubmed_abstract_checked` means **only** the publication context was checked, not any measured row.
2. Locate the compound in the source article or its supporting-information table. Record the **table/figure, supplement page, compound identifier and measured value**.
3. Verify the endpoint (biochemical target inhibition, cellular proliferation/viability, binding, etc.), explicitly tested protein construct (wild-type, named mutation, engineered sequence), assay format and units. Do not infer the construct from the paper title.
4. Check whether different BindingDB records are distinct assays, technical replicates, conversions from micromolar units, or duplicated publication reports. A duplicate signature is only a review hint.
5. Update the worksheet review fields. Keep `decision=defer_aggregation` until source evidence supports a reproducible inclusion/exclusion decision.

**Never use one paper's wild-type EGFR value as the label for a mutant EGFR construct or a cell-viability IC50 as an interchangeable purified-enzyme IC50.**

## Interpretation

This selection is not a representative sample. It deliberately prioritizes contradictory observations. It is suitable for demonstrating what kinds of assay heterogeneity appear in BindingDB and for motivating a conservative reconciliation policy; it does not quantify the incidence of laboratory bias in the full archive.

### How it connects to M1.10

The next milestone will specify a **versioned label reconciliation policy**: source checks, duplicate reporting, assay comparability, conflict thresholds, censored values, and sensitivity analyses. Even if some original tables are inaccessible, that limitation must result in `unverified` status or exclusion, not invented values or assumed assay equivalence.
