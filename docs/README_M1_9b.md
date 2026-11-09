# M1.9b — Prioritize repeated IC50 disagreements by molecule, target and provenance

**Status:** Descriptive triage; not final experimental reconciliation or training labels.

## Purpose

Investigate the 3,093 repeated molecule–target pairs identified in M1.9a, prioritizing:

1. Repeated **exact** IC50 records that disagree on the strict activity cutoff **IC50 < 1,000 nM**.
2. Pairs with reported IC50 spanning >10-fold (`range_pic50 > 1.0`).
3. Cases where study/provenance medians differ substantially and where disagreement occurs within a single publication/patent grouping.

This audit does not infer laboratory/site identity. A `provenance_key` generally corresponds to a PMID, DOI, or patent identifier; it does **not** identify an assay protocol, independent experimental replicate, or laboratory.

## Inputs and output locations

**Inputs (generated in M1.9a, keep private):**

- `data/interim/M1_9a_molecule_disagreement.csv` (one row per provisional structure ID and target)
- `data/interim/M1_9a_study_disagreement.csv` (one row per provisional structure ID, target and provenance key)

**Outputs:**

- `data/interim/M1_9b_priority_cases.csv` — all 3,093 repeated pairs, sorted by review tier and effect magnitude; keep in `data/interim/`.
- `data/interim/M1_9b_priority_studies.csv` — all provenance groups belonging to the 986 higher-priority pairs; keep in `data/interim/`.
- `reports/audits/M1_9b_disagreement_triage_summary.json` — reproducible counts, SHA-256 source hashes and top-10 review queue.
- `reports/figures/assay_disagreement/14_disagreement_triage.png` — descriptive visualization.

## Reproduce (Windows PowerShell, project root)

```powershell
uv run --locked pytest -q
uv run --locked ruff check scripts/triage_ic50_disagreements.py ml/tests/test_triage_ic50_disagreements.py

uv run --locked python .\scripts\triage_ic50_disagreements.py `
    --molecule-review .\data\interim\M1_9a_molecule_disagreement.csv `
    --study-review .\data\interim\M1_9a_study_disagreement.csv `
    --priority-cases .\data\interim\M1_9b_priority_cases.csv `
    --priority-studies .\data\interim\M1_9b_priority_studies.csv `
    --summary .\reports\audits\M1_9b_disagreement_triage_summary.json `
    --figure .\reports\figures\assay_disagreement\14_disagreement_triage.png
```

The script fails if molecule/group counts or summary statistics do not agree. It does not edit the M1.9a inputs, modify original IC50 values, or create any curated potency labels.

## Review tiers

| Tier | Decision rule | Consequence |
|---|---|---|
| T0 | Crosses 1000 nM activity boundary **and** >10-fold IC50 spread | Highest-priority source/assay investigation |
| T1 | Crosses activity boundary with ≤10-fold spread | Important for binary activity-label consistency |
| T2 | >10-fold spread but no threshold crossing | Important for regression-label consistency |
| T3 | Other repeated exact measurements | Review later / test reconciliation sensitivity |

**Important:** The thresholds prioritize manual review; they do not establish bad data. A pair can show >10-fold variation within one provenance group because a paper/patent can have multiple methods, assay formats, protein preparations or reporting entries.

## Preliminary descriptive findings from uploaded M1.9a exports

- Primary exact: 20,649 observations, 14,223 distinct molecule–target pairs.
- Repeated pairs: 3,093; threshold-crossing: 354; >10-fold: 904.
- Higher priority (union of threshold crossing or >10-fold): **986** pairs.
- Among the 904 >10-fold pairs, **599** have only one provenance key and **305** have multiple keys. This is **not** a within-laboratory versus between-laboratory comparison.
- **208** multi-provenance pairs have study-median spreads >1 pIC50 unit; **758** pairs exhibit >1 within at least one provenance group (overlap is possible).
- One severe EGFR example spans 0.24–83,000,000 nM among 21 observations across 18 provenance groups; this requires original-source verification.

## What remains before potency-label curation

M1.9b uses *aggregated* M1.9a outputs; they do not include each `reactant_set_id`, original assay protocol, ATP concentration, protein sequence, institution, or exact replicate status. This is a **review queue**, not completed causal or assay-level adjudication.

The next stage should join the priority pairs back to **M1.8c observation-level data**, then consult the underlying BindingDB fields/publications/patents where necessary. Preserve source identifiers, IC50 qualifiers, target constructs, and raw/standardized molecular IDs. Do not apply a single averaged label until observation-level curation and a documented conflict policy exist.
