# M1.8b — Identity discrepancy triage from uploaded audit files

**Status:** Source-versus-RDKit comparison; no curation, no corrected SMILES, no final labels.

Input: `M1_8a_identity_QA.csv`, SHA-256 `1153fa2d0cdb2e4fac02495eef4ffa847e523fb12c21e55b90cf308ccbf2a578`.

## Verified counts

| QA status (distinct raw SMILES) | Count |
| --- | ---: |
| Exact InChIKey match | 11,859 |
| Same InChI connectivity; source middle key `UHFFFAOYSA` | 6,019 |
| Same InChI connectivity; other differences in key layers | 15 |
| Different 14-character connectivity block | 30 |
| RDKit key unavailable / parsing failure | 2 |
| **Total** | **17,925** |

Source identity disagreements are not proof of wrong SMILES. Of the 30 connectivity block mismatches, 25 unique structures are encountered in US Patent rows and five in ChEMBL rows. The two parse failures occur in three observation records (IDs `695385`, `695386`, and `50695132`). One SMILES has multiple dot-delimited components, and should not be automatically salt-stripped.

## High-priority examples

- `117755`: source key `FVDRAHRSWLZRPW-UHFFFAOYSA-N`; computed key `QDAVLQQAHAKDBT-FYYLOGMGSA-N`. The first (connectivity) block differs. Requires source-document/chemical identity investigation.
- `695385` and `695386`: the same SMILES contains a bracketed nitrogen whose RDKit sanitization fails an explicit-valence check. This does not justify inventing a replacement SMILES.
- `50695132`: the supplied string includes CXSMILES-like `|TLB:...|` annotations. In a local diagnostic using RDKit 2025.09.4, removing the trailing annotation permitted parsing of its core SMILES; this diagnostic is *not* permission to silently discard source annotations, and the user's original audit uses RDKit 2026.09.1.

## Repeated identifiers

- 653 **source-reported** InChIKeys map to multiple RDKit canonical isomeric SMILES in the input.
- Eight **RDKit-calculated** full InChIKeys map to multiple canonical isomeric SMILES.

This is why we must not use any single source-reported key as an unqualified unique compound identifier. Some canonical SMILES differ due to representation details even when InChIKey is equal; a documented grouping policy is needed.

## Model-building implications

1. Keep original SMILES, source key, calculated key, and the QA flags.
2. For the primary exact-IC50 pool, do not group compounds based on the first InChIKey block alone. That could collapse distinct stereoisomers.
3. Quarantine (not delete) unparseable structures and connectivity discrepancies from *automatic* identity reconciliation until reviewed.
4. Check salt/fragment content, aromaticity, charge, protonation, stereochemistry completeness, and tautomer conventions explicitly before choosing any standardized identifier.
5. Rejoin molecule identity to assay observations using a verified stable record ID before analysis of pIC50 variation, publication bias, or mutant vs reference potency shifts.
6. Do not calculate model splits or group repeated assay results until the identity policy is documented.

No independent experimental potency values or causal lab effects were established by this QA.
