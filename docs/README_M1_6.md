# M1.6 — Construct name + publication/patent provenance diagnostics

Extract **the contents of this ZIP into your project root**, merging `scripts/`, `ml/tests/`, and `reports/`. It does not replace the prior M1.4 or M1.5 code.

## Validate

```powershell
uv run --locked pytest -q
uv run --locked ruff check scripts/audit_construct_provenance.py ml/tests/test_audit_construct_provenance.py
```

## Reproduce from your corrected M1.4 observations

```powershell
uv run --locked python .\scripts\audit_construct_provenance.py `
    --input .\data\interim\bindingdb_egfr_her2_observations.csv `
    --summary .\reports\audits\M1_6_construct_provenance_profile.json `
    --inventory .\reports\audits\M1_6_target_name_inventory.csv `
    --figure .\reports\figures\source_bias\08_construct_name_categories_by_source.png
```

The supplied outputs were generated from the 35,210-row M1.4 observation CSV supplied in the conversation. Your locally regenerated outputs become the canonical artifacts for your repository.

### Interpretation

See `reports/audits/M1_6_interpretation.md`. All metrics are **name- and provenance-based diagnostics, not sequence-verified protein identities or labels**. No drug discovery model is trained here.
