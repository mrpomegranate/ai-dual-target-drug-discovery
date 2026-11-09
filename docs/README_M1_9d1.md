# M1.9d.1 — Original-source review queue

**Status:** Descriptive evidence reconstruction and review triage; **no label aggregation or model training**.

## Purpose

Previous milestones detected substantial discrepancies among experimentally reported IC50 values for the same provisional chemical structure and nominal EGFR/HER2 target. M1.9d.1 creates a reproducible, observation-validated queue for checking **individual assays and original source tables**. It does not equate a publication with an individual laboratory, and it does not assume the unannotated BindingDB target name proves wild-type sequence or a biochemical kinase-inhibition endpoint.

## Inputs

- `data/interim/M1_9c_priority_observation_evidence.csv` (1,001 raw exact-IC50 observations in 20 selected extreme cases)
- `data/interim/M1_9c_priority_provenance_groups.csv` (656 publication/patent groups)

These are *purposively selected* high-disagreement cases, not a random or representative sample.

## Run from the repository root (Windows PowerShell)

```powershell
uv run --locked pytest -q
uv run --locked ruff check scripts/build_source_review_queue.py ml/tests/test_source_review_queue.py

uv run --locked python .\scripts\build_source_review_queue.py `
    --evidence .\data\interim\M1_9c_priority_observation_evidence.csv `
    --provenance .\data\interim\M1_9c_priority_provenance_groups.csv `
    --queue .\data\interim\M1_9d1_source_review_queue.csv `
    --summary .\reports\audits\M1_9d1_source_review_summary.json `
    --figure .\reports\figures\assay_disagreement\16_priority_metadata_coverage.png
```

## Review priorities

- **Highest:** same provisional structure + target + provenance key has exact measurements on both sides of the strict 1,000 nM activity threshold.
- **High:** within that grouping the exact IC50 range is greater than tenfold (pIC50 range >1), without a threshold contradiction.
- **Routine:** other source groups in the selected 20-case review cohort.

Priorities overlap conceptually; a threshold-crossing group is counted only in `highest` for the queue, even if its spread is also >10-fold.

## Scientific investigation

For each high-priority group, compare each BindingDB reactant-set ID and molecule identifier against the original article/patent and its supplementary tables. Specifically check (a) measured endpoint: purified kinase inhibition vs cellular proliferation/viability vs phosphorylation or binding; (b) actual target protein construct and mutation; (c) concentration units and inequality/qualifier; (d) assay context such as ATP, substrate, cell line; (e) duplicated reporting and source reuse. Document the source table/page and your confidence.

Do not mark an observation as confirmed simply because the paper title or abstract mentions a target or mutation; it must be **matched to a specific original experimental entry**. Do not drop discrepant values or compute an averaged training label in this stage. Complete the queue's empty review columns after verification.

## Outputs

- `data/interim/M1_9d1_source_review_queue.csv` — manual investigation queue. **Keep private / out of public Git.**
- `reports/audits/M1_9d1_source_review_summary.json` — counts + input SHA256 hashes, safe summary.
- `reports/figures/assay_disagreement/16_priority_metadata_coverage.png` — visualization of missing assay metadata.
- `reports/audits/M1_9d1_original_source_leads.md` — checked publication descriptions and review cautions.

## Completion criteria

A complete M1.9d source investigation requires record-to-assay/construct matching for at least a justified set of priority cases and transparent exclusions for unresolvable assays. A reproducible review *queue* is an intermediate deliverable, **not evidence that the publication's original assay values have been verified**.
