# M1.8d — Molecular representation collision review

**Status:** A reproducible implementation and preliminary run on the shared M1.8c output. The definitive results are the user's local run after passing the repository tests.

## Scientific question

If multiple raw SMILES map to the same canonical isomeric SMILES, or if multiple canonical SMILES share one recalculated InChIKey, should we merge them? **Not by default.** These two cases need different handling.

- **Raw SMILES variants -> same canonical isomeric SMILES:** the current M1.8c exact-structure identity already joins these representations; preserve original strings and provenance separately. The observation labels still must not be averaged without assay review.
- **One full recalculated InChIKey -> multiple canonical isomeric SMILES:** keep distinct M1.8c structure IDs. Some cases involve tautomer-like changes or alternative charged/un-charged representations. Do not equate assays or merge IC50 labels. Flag their shared identifier for potential split leakage when building scaffold-held-out sets.

## Preliminary observations

From the uploaded M1.8c result, 35,210 observations were retained, 35,127 approved provisional structure identities, and 83 review holds. The M1.8d script found:

| Collision type | Groups | Observation rows | Interpretation |
|---|---:|---:|---|
| Multiple raw SMILES -> same canonical SMILES | 4 | 154 | Canonical representations already joined by M1.8c |
| One recalculated InChIKey -> multiple canonical SMILES | 8 | 41 | Keep structure IDs distinct; investigate potential equivalence families and leakage |

These are groups and observation rows, **not 12 mutually exclusive molecules**, and the two sets may overlap. No IC50 values are modified.

## Files

- `scripts/review_structure_collisions.py`
- `ml/tests/test_review_structure_collisions.py`
- `reports/audits/M1_8d_collision_summary.json`
- `reports/audits/M1_8d_collision_review.csv`
- `reports/figures/molecular_identity/12_collision_groups_and_observations.png`

## Reproduce from repository root (PowerShell)

Before this step, run M1.8c and ensure that its standardized observation table and collision inventory correspond to the **same** input version.

```powershell
uv run --locked pytest -q
uv run --locked ruff check scripts/review_structure_collisions.py ml/tests/test_review_structure_collisions.py

uv run --locked python .\scripts\review_structure_collisions.py `
    --input .\data\interim\M1_8c_standardized_observations.csv `
    --inventory .\reports\audits\M1_8c_representation_collisions.csv `
    --review .\reports\audits\M1_8d_collision_review.csv `
    --summary .\reports\audits\M1_8d_collision_summary.json `
    --figure .\reports\figures\molecular_identity\12_collision_groups_and_observations.png
```

## QC and limitations

- Fail on repeated or missing Reactant Set IDs, missing required columns, incomplete approved identities, inconsistent canonical SMILES-to-structure-ID mappings, or a stale collision inventory.
- No chemistry normalization, no InChIKey-level automatic structure merging, no assays or IC50s aggregated, no train/validation/test split.
- The case inventory displays full chemical structure strings; review public-repo sharing requirements before `git add`.
- Related representation families must be isolated across future splits as appropriate, **in addition** to controlling molecular-scaffold overlap and publication/assay leakage.
- Construct names were not sequence-verified; the reference-like dataset remains provisional.

## Next stage

M1.9: examine repeated observations for each `(provisional structure ID, target, construct)` and their spread in exact pIC50 **by publication and assay context**. Build a reconciliation policy before generating final potency labels. Where assay identifiers are absent, report the limitation instead of equating one patent to one laboratory.
