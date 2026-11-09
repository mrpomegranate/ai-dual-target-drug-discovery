# M1.9a — Exact-IC50 disagreement and provenance diagnostics

**Status:** Implementation preview (run against the latest local M1.8c standardized observations to obtain the authoritative report).

## Why this milestone exists

The molecular-identity audits (M1.8a–d) established provisional identities for EGFR/HER2 inhibitor compounds. They did **not** show that two experiments using the same molecule have equivalent assay conditions or provide interchangeable IC50 labels.

M1.9a asks: **How much do repeated exact IC50 measurements disagree when grouped by provisional chemical identity and target?** It also asks how much that variation is associated with publication/patent grouping and BindingDB source categories.

This is observational QA, **not model evaluation, not causal laboratory bias estimation, and not final label aggregation**.

## Scientific population and definitions

- Start from the 35,210-row `M1_8c_standardized_observations.csv`.
- Include only rows where `m17_partition == primary_exact_candidate` and `m18c_status == provisional_exact_structure`.
- Retain only exact IC50 with a consistent, positive numeric `ic50_nm` and pIC50 = `9 - log10(IC50_nM)`.
- A **molecule–target pair** is `(target, m18c_structure_id)`. EGFR and HER2 never become one label.
- A **provenance group** is `(target, m18c_structure_id, provenance_key)`, using the upstream PMID/patent/DOI rule. A provenance group is **not an assay identifier, individual laboratory, or independently verified replicate**.
- For pairs with ≥2 observations, `range_pic50 = max(pic50) - min(pic50)`; a range greater than `1.0` means more than a tenfold range in reported IC50.
- A **source-group contrast** compares medians of different BindingDB `source` categories for the *same molecule–target pair*. It does not remove variation in assay protocols or represent causal source bias.

## Reproduce from project root (PowerShell)

```powershell
uv run --locked pytest -q
uv run --locked ruff check scripts/audit_ic50_disagreement.py ml/tests/test_audit_ic50_disagreement.py

uv run --locked python .\scripts\audit_ic50_disagreement.py `
    --input .\data\interim\M1_8c_standardized_observations.csv `
    --summary .\reports\audits\M1_9a_disagreement_summary.json `
    --molecule-review .\data\interim\M1_9a_molecule_disagreement.csv `
    --study-review .\data\interim\M1_9a_study_disagreement.csv `
    --figure .\reports\figures\assay_disagreement\13_primary_exact_ic50_disagreement.png
```

Check the input SHA-256 and run outputs, and upload only `reports/audits/M1_9a_disagreement_summary.json` for first review. Keep the full molecule/study-level CSVs under `data/interim/` (gitignored) unless specifically needed for closer analysis.

## Preliminary findings on the uploaded data

The following **preview** came from `M1_8c_local_standardized_observations.csv` with SHA-256:
`dc6e131c8fc1f242b93fa885f540809f28934808cc8611e26a3bd96bd7be6e7d`.

| Metric | Value |
|---|---:|
| Full original observation rows | 35,210 |
| Eligible exact-IC50 observations with provisional identity | 20,649 |
| Distinct molecule–target pairs | 14,223 |
| Pairs with repeated exact observations | 3,093 |
| Repeated pairs with >1 pIC50 range (>10-fold IC50) | 904 |
| Pairs spanning ≥2 provenance groups | 1,351 |
| Pairs in ≥2 BindingDB source categories | 431 |
| Matched-source pairs with >1 pIC50 source-median range | 55 |

Only observations and provisional structure IDs are counted, **not validated independent assays or finalized training examples**.

## Interpret correctly

- A **tenfold range** is a high-priority disagreement flag, not proof that a study is wrong or biased.
- Many repeated rows can represent technical replicates, distinct assay formats, variant/construct differences not reflected in the plain target name, or republication/duplicate ingestion of the same result.
- Same IC50 in the same provenance group is a **possible duplicate-reporting signature**, not confirmation of duplication.
- Min–max disagreement is sensitive to sample size and outlying measurements. Study median contrasts are descriptive only.
- The percentage of cases crossing the 1,000 nM threshold should be reported alongside regression disagreement.
- The current data are inadequate to infer *laboratory-specific causal bias*: no verified laboratory IDs, no uniform assay conditions, and limited pH/temperature metadata.
- DO NOT aggregate pIC50 or generate final training labels in this stage.

## Next decision point

M1.9b should investigate top-disagreement examples, assay-record identifiers, study/source provenance, and whether some rows report the same experimental measurement across databases. Only after that should a documented label policy be considered, including handling conflicting/censored evidence and an uncertainty flag.
