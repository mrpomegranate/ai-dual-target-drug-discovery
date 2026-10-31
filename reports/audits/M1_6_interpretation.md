# M1.6 — Protein construct and provenance audit (diagnostic only)

## Input and output

- **Input:** M1.4 observation-level `data/interim/bindingdb_egfr_her2_observations.csv`.
- **Script:** `scripts/audit_construct_provenance.py`.
- **Outputs:** `reports/audits/M1_6_construct_provenance_profile.json`, `reports/audits/M1_6_target_name_inventory.csv`, and `reports/figures/source_bias/08_construct_name_categories_by_source.png`.
- **Scope:** Does not modify source observations, identify laboratories, align protein sequences, standardize molecules, merge assay records or generate training labels.

## Main results on supplied 35,210-row observation CSV

| Name class | Observations | Interpretation |
|---|---:|---|
| Plain target name | 25,733 | No construct information specified in the target name; **not automatically wild-type**. |
| Substitutions only | 4,524 | Point substitutions specified in target name, such as `[L858R]` or `(L858R)(T790M)`. |
| Substitutions + other construct changes | 2,080 | Mutation plus deletion/truncation/range or insertion notation; require manual/sequence review. |
| Other construct annotations | 2,873 | Bracket contents such as sequence ranges or inserted residues; no unambiguous simple point-substitution-only signature. |
| Unclassified non-base names | 0 | No unhandled formats in this file. |

**Correction to M1.5:** 100 EGFR observations use the parentheses style `(EGFR)(L858R)(T790M)` instead of square brackets. All 100 were from the **US Patent** curation source, not ChEMBL. Thus the original `bracket_annotation_percent` metric measures bracket notation only and should not be presented as the complete construct-annotation rate. Figure 08 supplements the earlier figure 02 and is preferred for name-format-aware diagnostics.

**Source-reported InChIKey overlap** (not curated labels): 2,400 identifiers occur under both targets. EGFR has 2,188 InChIKeys with multiple construct *categories*; HER2 has 443. Even an identical InChIKey does not authorize averaging outcomes from different target constructs, experimental sources, or censoring schemes.

## Provenance is not laboratory identity

The script chooses a **patent number**, otherwise **PMID**, otherwise **article DOI**, as a source grouping proxy when available. The grouping variable can identify associated publications or patent documents, but does not prove that all assays within a publication or patent used the same laboratory, experimental conditions, or biological construct. In the user's extracted file, all records have at least one of these identifiers, but this is *identifier presence* rather than assay protocol completeness.

## Recommended next scientific checks

1. Retrieve **target chain sequences** (and sequence accession/construct details) from the original 640-column source for selected name classes before assuming wild-type identity.
2. Stratify repeated molecule–target observations by exact construct name/sequence, study identifiers, and assay metadata. Quantify differences in exact pIC50 across studies *after* construct checks.
3. Implement transparent RDKit molecular standardization and a molecule-level reconciliation policy; reserve unmatched/censored evidence without inventing point estimates.
4. Compare RF and GNN on scaffold-held-out testing, with source-held-out sensitivity analysis as secondary robustness evaluation.
