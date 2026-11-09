# M1.8b — Investigate molecular identity discrepancies

**Status:** Reproducible source-versus-RDKit QA. **Not** a molecular standardization or aggregation step.

## Purpose

Review discrepancies discovered in M1.8a before choosing a chemical identity policy for EGFR/HER2 potency labels.

- Exact full-key matches need no key-disagreement review (but remain subject to assay QA).
- Same 14-character InChI connectivity block with different remaining characters may reflect specified stereochemistry or other differences in representation; **do not automatically merge or drop**.
- Different InChI connectivity blocks receive **high-priority manual review**, without a presumption that either representation is erroneous.
- RDKit parse failures receive **high-priority review**; original SMILES must be retained.
- Multiple fragments, and distinct canonical isomeric SMILES sharing one InChIKey, require separate attention before grouping molecules.

## Reproduce

Run from the repository root:

```powershell
uv run --locked pytest -q
uv run --locked ruff check scripts/review_identity_discrepancies.py ml/src/dual_target/identity_review.py ml/tests/test_identity_review.py
uv run --locked python .\scripts\review_identity_discrepancies.py `
    --input .\data\interim\M1_8a_identity_QA.csv `
    --review .\data\interim\M1_8b_identity_review.csv `
    --priority .\data\interim\M1_8b_priority_review.csv `
    --summary .\reports\audits\M1_8b_identity_discrepancy_summary.json
```

## Outputs

- `data/interim/M1_8b_identity_review.csv`: one row per raw SMILES, including status, reported keys, computed keys, source and target membership.
- `data/interim/M1_8b_priority_review.csv`: high-priority rows for **manual inspection**; keep this observation-derived chemistry data out of the public Git repository.
- `reports/audits/M1_8b_identity_discrepancy_summary.json`: aggregate QA counts, safe to share as counts.

## Scientific and modeling implications

For the initial molecule-only GNN/RF comparison, a defensible candidate *strict* identity key is RDKit canonical **isomeric** SMILES, with RDKit full InChIKey as a cross-check, and no silent salt stripping, deprotection, stereoisomer collapse, tautomerization, or protonation normalization. RDKit canonical SMILES itself does not prove physical identity, stereochemistry completeness, or chemical equivalence of experimental assay material. An explicit hierarchy and stereochemistry/salt policy should be decided next, after reviewing flagged structures.

**No observations are removed or relabeled at this step.** No IC50 labels are averaged. Never interpret one InChIKey per canonical SMILES as guaranteed; at least some distinct SMILES can have the same full InChIKey.

For the main benchmark, model evaluation must be scaffold-disjoint. Keep the EGFR/HER2 construct challenge separate and audit molecule and scaffold overlap after standardization.

## Critical limitations

1. The `M1_8a_identity_QA.csv` contains chemical-identity fields **but not the underlying IC50 measurements or publication DOI/patent number**. Rejoin to M1.7/source observations by stable `reactant_set_id` when assessing assay disagreements; validate join uniqueness first.
2. The current triage cannot establish which source representation is correct without inspection of literature/patent structures and upstream metadata.
3. The source InChIKey's `UHFFFAOYSA` middle block is consistent with lack of specified stereochemistry, but the entire mismatch cannot be assumed to be solely stereochemistry without inspecting the underlying structures.
4. M1.8b alone does not produce curated dual-target compound counts or train/test splits.
