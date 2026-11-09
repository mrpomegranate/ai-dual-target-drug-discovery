# M1.8c — Conservative identity audit: interpretation

**Data examined:** M1.7 provisional decisions joined row-for-row with the M1.8a RDKit audit and the M1.8b high-priority structure list.

## Reconciliation

| Structural decision | Observation rows | Interpretation |
|---|---:|---|
| Provisional exact-structure ID assigned | **35,127** | RDKit-parsed, one fragment, not among flagged connectivity disagreements; **not final curated molecules** |
| Hold: reported/RDKit connectivity disagreement | **79** | 30 distinct SMILES, retained with no approved ID |
| Hold: RDKit parse failure | **3** | 2 distinct SMILES, retained with no approved ID |
| Hold: multi-fragment structure | **1** | One additional structure withheld from automatic matching; no salt stripping or component selection |
| **All observations** | **35,210** | All original evidence preserved |

The 32 unique high-priority SMILES correspond to **82 observations** (79 connectivity disagreement and 3 parse-failure rows). The one multi-fragment hold is a separate record.

By original M1.7 partition:

| M1.7 partition | Provisional ID assigned | Held | Total |
|---|---:|---:|---:|
| Primary exact IC50 | 20,649 | 16 | 20,665 |
| Primary censored classification | 4,844 | 1 | 4,845 |
| Secondary construct reserve | 9,421 | 56 | 9,477 |
| Primary measurement holds | 213 | 10 | 223 |

The **provisional same-ID EGFR/HER2 overlap** in primary exact observations is **1,662**. This is not the number of assay-reconciled, dual-target molecules, nor a train/test split.

## Additional representation diagnostics

- Four canonical isomeric SMILES each represent more than one raw SMILES spelling.
- Eight RDKit InChIKeys correspond to multiple canonical isomeric SMILES. The canonical SMILES hash, **not** InChIKey, is the provisional join key; this does not silently collapse those structures.
- Exact stereochemistry is kept when present. Missing stereochemical specification is flagged and **not filled in**.
- Current figures and summary generated for this preview used RDKit **2025.09.4** for stereo flags, while upstream M1.8a canonicalization was computed under RDKit **2026.09.1**. Regenerate the report locally using the repository's locked environment before treating it as authoritative. IDs are derived from the upstream M1.8a canonical SMILES and are not recalculated here.

## Scientific safeguards and next step

The observations still contain different target constructs, patent/publication sources and possibly discordant assay conditions. Structural identity alone cannot authorize IC50 averaging. Before dataset assembly, review the 12 structural representation collision groups and investigate duplicate study/assay measurements, conflicting pIC50, left/right-censoring, and study-held-out sensitivity. The protein-construct challenge pool remains reserved and must be evaluated separately from primary scaffold-held-out generalization.
