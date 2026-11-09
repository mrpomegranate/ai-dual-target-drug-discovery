# M1.9c — Individual observation and provenance evidence for priority IC50 disagreements

**Status:** reproducible evidence extraction and metadata QA; **not** experimental-error attribution, potency label reconciliation, or model training.

## Why this checkpoint matters

The M1.9b triage identified 986 priority EGFR/HER2 molecule–target combinations, including 354 cases with contradictory activity classifications and 904 with more than tenfold pIC50/IC50 disagreement (the sets overlap). These are measured differences, **not** proof that any laboratory is biased. An observation can differ because of protocol, protein construct, ATP concentration, experimental variability, duplication/reporting, or data issues. Many assay fields are missing.

M1.9c reconstructs original **exact** IC50 observations from the M1.8c observation table using the M1.9b `target + m18c_structure_id` keys, excluding censored measurements and annotated constructs. It deliberately selects the 10 top-ranked *EGFR* cases and the 10 top-ranked *HER2* cases so smaller HER2 cohorts are not crowded out by EGFR. This sample is biased toward extremes and is **not representative** of the complete training dataset.

## Inputs and provenance

- `data/interim/M1_8c_standardized_observations.csv` — observation-level evidence with original BindingDB identifiers, raw IC50, metadata and provisional molecular IDs.
- `data/interim/M1_9b_priority_cases.csv` — M1.9b ranked review queue.
- `reports/audits/M1_9a_disagreement_summary.json` — optional but **recommended** for the standardized input SHA-256 verification.

The script independently validates the exact-IC50 policy, pIC50 formula, strict IC50 `<1000 nM` classification, provenance counts, selected-case observation counts, minimum/maximum IC50, and pIC50 ranges. If any fail, it raises an error rather than outputting a misleading report.

## Commands (PowerShell; from repository root)

```powershell
uv run --locked pytest -q
uv run --locked ruff check scripts/audit_priority_assay_evidence.py ml/tests/test_audit_priority_assay_evidence.py

uv run --locked python .\scripts\audit_priority_assay_evidence.py `
    --input .\data\interim\M1_8c_standardized_observations.csv `
    --cases .\data\interim\M1_9b_priority_cases.csv `
    --m19a-summary .\reports\audits\M1_9a_disagreement_summary.json `
    --per-target 10 `
    --evidence .\data\interim\M1_9c_priority_observation_evidence.csv `
    --study-summary .\data\interim\M1_9c_priority_provenance_groups.csv `
    --summary .\reports\audits\M1_9c_assay_evidence_summary.json `
    --figure .\reports\figures\assay_disagreement\15_priority_case_assay_evidence.png
```

## Outputs

| Location | Description | Commit? |
|---|---|---|
| `data/interim/M1_9c_priority_observation_evidence.csv` | Original individual BindingDB measurements for selected cases; includes structure strings and source IDs | **No** (keep private/local) |
| `data/interim/M1_9c_priority_provenance_groups.csv` | Selected case × publication/patent summaries, descriptive only | **No** (keep private/local) |
| `reports/audits/M1_9c_assay_evidence_summary.json` | Aggregate metadata coverage, selection and checksum information | Yes, after verification |
| `reports/figures/assay_disagreement/15_priority_case_assay_evidence.png` | Raw pIC50 ranges for top priority cases from both targets | Yes |

## Preliminary output from uploaded analysis data

The local script was exercised against the previously uploaded M1.8c observation file and M1.9b priority case table, yielding **20 selected cases, 1,001 exact observation rows, 656 publication/patent provenance groups, 62 rows sharing a possible duplicate-reporting signature**, and **115 provenance groups with >1.0 pIC50 range**. Only 12 of the 1,001 rows had populated pH metadata and 12 had temperature metadata. Re-run in your locked environment; the local summary, checksum and figure are authoritative after verification.

**Important interpretation constraints:**

- One provenance group is not necessarily one assay, lab, or independent measurement study.
- Exact matching of source, target, numerical value and available context may represent duplicated reporting **or** experimental replicates. No duplicates are dropped here.
- Missing pH, temperature, protein sequence, ATP concentration and detailed protocol prevent controlled causal claims about laboratory/source effects.
- Study medians are for descriptive evidence display; they are **not** molecule–target training labels.
- There is no inference of abnormal or erroneous labels based solely on potency range. An extreme value remains part of evidence until independently checked.

## Next checkpoint

M1.9d should audit original assay descriptions for a focused selection of contradictory examples, and then codify transparent options for label reconciliation (e.g. exclude unresolved extreme disagreements; study-aware sensitivity analyses). Keep all raw observations immutable and produce explicit exclusion reasons. Finally build scaffold-disjoint splits and only then train and evaluate Random Forest and GNN baselines.
