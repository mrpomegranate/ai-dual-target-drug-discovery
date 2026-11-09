# M1.9d.2 — Casebook verification and source-context addendum

**Scope:** 44 selected EGFR/HER2 experimental observations; 14 molecule–target–publication/patent groups.

## Independent consistency checks

- 44 rows; all reactant-set IDs unique.
- 14 group keys; counts sum to 44.
- Each worksheet group matches its underlying observation count, reactant-set ID membership, min/max IC50 and pIC50 range.
- All 14 groups currently defer aggregation. No row-to-source-table match has been demonstrated.

## Highest-priority groups

| Target | Provenance | Rows | IC50 range (nM) | Decision |
|---|---|---:|---:|---|
| EGFR | `pmid:32931277` | 4 | 0.02–10117 | Hold: source-table verification pending |
| EGFR | `pmid:32931277` | 5 | 0.06–14781 | Hold: source-table verification pending |
| EGFR | `pmid:32931277` | 5 | 0.01–1401 | Hold: source-table verification pending |
| EGFR | `pmid:32361329` | 3 | 46–377400 | Hold: source-table verification pending |
| EGFR | `pmid:32361329` | 3 | 95–421000 | Hold: source-table verification pending |
| EGFR | `pmid:28974338` | 4 | 12–22800 | Hold: source-table verification pending |
| HER2 | `pmid:18077363` | 2 | 3–3400 | Hold: source-table verification pending |
| HER2 | `pmid:18182285` | 2 | 3–1000 | Hold: source-table verification pending |
| HER2 | `pmid:16111887` | 3 | 30–5800 | Hold: source-table verification pending |
| HER2 | `patent:US10143695` | 5 | 9.4–1000 | Hold: source-table verification pending |
| HER2 | `pmid:32739648` | 2 | 71–1230 | Hold: source-table verification pending |
| HER2 | `patent:US11807649` | 2 | 422–1930 | Hold: source-table verification pending |
| EGFR | `pmid:37197473` | 3 | 0.029–313 | Hold: source-table verification pending |
| EGFR | `pmid:36375337` | 1 | 8.3e+07–8.3e+07 | Hold: source-table verification pending |

## External publication context — NOT row-to-table matches

- **pmid:32931277** (Publication context verified): Wild-type HER2 and HER2 insertion mutant considered; paper lists IC50 and EC50 supplemental data. Individual EGFR assay record linkage unresolved. Source: https://pubmed.ncbi.nlm.nih.gov/32931277/ ; https://pubs.acs.org/doi/10.1021/acs.jmedchem.0c00870
- **pmid:32361329** (Publication abstract verified): Study discusses cell-line cytotoxicity (HCT116, HepG2, MCF-7), EGFR inhibition and apoptosis; row-specific assay endpoint unresolved. Source: https://pubmed.ncbi.nlm.nih.gov/32361329/ ; https://doi.org/10.1016/j.ejmech.2020.112363
- **pmid:37197473** (Full text verified): Paper reports EGFR WT/L858R/L858R-C797S biochemical HTRF and mutant-dependent cellular CTG assays; row-specific construct/endpoint unresolved. Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC10184309/ ; https://doi.org/10.1021/acsmedchemlett.2c00514
- **pmid:36375337** (Publication title verified): Paper investigates mutant EGFR and cancer drug discovery; extraordinarily high 83,000,000 nM observation requires primary-source unit/endpoint verification. Source: https://pubmed.ncbi.nlm.nih.gov/36375337/ ; https://doi.org/10.1016/j.ejmech.2022.114889
- **pmid:28974338** (Publication title verified): WZ4002 analogue EGFR inhibitors; actual assay construct for selected repeated values unresolved. Source: https://pubmed.ncbi.nlm.nih.gov/28974338/ ; https://doi.org/10.1016/j.bmcl.2017.09.048

## Interpretation and curation safeguards

- A source publication can include multiple protein constructs, assay platforms and biochemical vs cellular endpoints. A generic BindingDB target name is insufficient to certify assay equivalence.
- Do not infer that a source paper having an EGFR assay proves a specific row is biochemical EGFR or wild type.
- Preserve original values, linkage keys, source provenance and M1.7/M1.8 eligibility flags.
- Until row-to-table verification, assign no assay-specific correction, deletion, mutation label, or aggregate pIC50.
- The 14 groups are a purposively selected subset; do not extrapolate their error or ambiguity rates to the full dataset.

## Recommended next action for M1.10

Design conservative primary-label rules and a sensitivity analysis. For ambiguous high-disagreement combinations, use `hold_unresolved_assay` rather than silently averaging. Continue independent original-assay verification where feasible, and report its scope.

## Local files reconciled

- `M1_9d2_selected_source_observations.csv`: SHA256 `cd0d7d88a1a7b42b4d35499901ab3d3cc16c8ce6c56f25cde1cd592a543f7b7a`
- `M1_9d2_source_verification_worksheet.csv`: SHA256 `7a1266eec2af70bc7496850587b75ea0c0e985f5659d3fe1c2d3b170ce42544f`
