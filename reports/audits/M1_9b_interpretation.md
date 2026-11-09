# M1.9b — Interpretation of IC50 disagreement triage

**Evidence:** User-uploaded M1.9a molecule- and study-provenance summary CSVs (20,649 exact IC50 observations, 14,223 molecule–target pairs); the authoritative local rerun should verify input hashes.

## Review queue

| Tier | Pairs | Definition |
|---|---:|---|
| T0 — contradictory classification + >10-fold | 272 | Both active and inactive exact observations; pIC50 range >1 |
| T1 — contradictory classification, ≤10-fold | 82 | Both active and inactive exact observations; pIC50 range ≤1 |
| T2 — >10-fold but same classification | 632 | Large potency disagreement while not straddling the strict 1,000 nM cutoff |
| T3 — other repeated pairs | 2,107 | Repeated exact records not meeting the above review priorities |
| **Total repeated** | **3,093** | 986 in T0–T2 priority queue |

Observed >10-fold differences may arise from assay context, repeated reporting, biological construct variation, metadata inconsistencies, data-entry/units issues or genuine measurement variability. None is established by this aggregate analysis.

Among 904 >10-fold pairs, 599 occur under **one** provenance identifier, 305 under **multiple** identifiers. A provenance identifier is a publication/patent grouping, **not** a lab or experimental assay identifier.

**758 distinct pairs** include a single provenance grouping whose within-group spread exceeds 1 pIC50 unit. **208 pairs** spanning multiple provenance groups have a spread of study medians exceeding 1 unit. Of those properties, **69 pairs satisfy both**, so they should not be added as disjoint categories.

## High-priority example (from actual uploaded aggregate data)

An EGFR structure ID beginning `RDKISO1-33c11ca...` has 21 exact IC50 observations from 18 provenance groups, spanning **0.24 nM to 83,000,000 nM** (8.539 pIC50 units). This is an **outlier investigation priority**, not evidence that a given lab is inaccurate. The compound name, source assay conditions, and individual measurement identifiers have not yet been retrieved in this phase.

## Decision before model training

- Keep both original exact measurements and source provenance in the audit trail.
- Flag the T0–T2 pairs for assay-level investigation and later label-policy sensitivity analyses.
- **Do not** infer an institution/laboratory batch effect from provenance groups alone.
- **Do not** treat study medians as training labels.
- **Do not** claim independent wet-lab validation or external test performance.

Recommended next checkpoint: reconstruct the original observations for the priority cases, audit assay metadata and potential duplicated reporting, then propose transparent label reconciliation with explicit uncertainty/conflict exclusions.
