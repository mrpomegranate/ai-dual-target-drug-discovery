# M1.8c — Conservative molecular standardization and provisional identity assignment

**Status:** Prepared implementation; locally validated only when the user runs the supplied tests and full audit.  
**Scope:** EGFR/HER2, October 2026 BindingDB; preserve all M1.7 provisional construct partitions.

## Scientific purpose

Convert M1.8a RDKit structure QA and M1.8b identity-discrepancy flags into a reproducible **provisional structure-identity policy**. This does **not** reconcile multiple IC50 measurements and does not designate final train/validation/test molecules.

### Policy

1. Preserve **every original M1.7 observation and all source identifiers**, target constructs, publication/patent provenance, IC50 qualifiers, and pIC50 values. No assay information is modified.
2. Use the **RDKit canonical isomeric SMILES already calculated in M1.8a** as the provisional exact-structure representation, only when parseable and single-fragment. `RDKISO1-<sha256>` is a policy-versioned hash of the entire canonical isomeric SMILES, **not** of the source-reported InChIKey.
3. **Quarantine** M1.8b high-priority structures: 30 distinct SMILES with connectivity-block disagreement and two RDKit parse failures. All related observations remain present, but do not get an approved provisional structure ID.
4. Separately hold multi-fragment structures without stripping salts or choosing a largest fragment.
5. **Do not** merge stereoisomers, assign unspecified stereochemistry, collapse tautomer states, neutralize charges, remove counterions, or rewrite raw SMILES. Flag potentially stereogenic elements left unspecified in supplied structures as a diagnostic, not an exclusion criterion.
6. Two observations having the same provisional structure ID **does not mean** their target constructs, assay protocols, or IC50 values can be averaged.
7. Fail closed on mismatched M1.7/M1.8a identifiers, missing record IDs, or a stale M1.8b priority list.

### Inputs

```text
data/interim/M1_7_observation_decisions.csv
data/interim/M1_8a_identity_QA.csv
data/interim/M1_8b_priority_review.csv
```

### Validate

```powershell
uv run --locked pytest -q
uv run --locked ruff check scripts/apply_molecular_standardization.py ml/src/dual_target/standardization.py ml/tests/test_standardization.py
```

### Run (PowerShell, repository root)

```powershell
uv run --locked python .\scripts\apply_molecular_standardization.py `
    --input .\data\interim\M1_7_observation_decisions.csv `
    --identity .\data\interim\M1_8a_identity_QA.csv `
    --priority .\data\interim\M1_8b_priority_review.csv `
    --output .\data\interim\M1_8c_standardized_observations.csv `
    --summary .\reports\audits\M1_8c_standardization_summary.json `
    --collisions .\reports\audits\M1_8c_representation_collisions.csv `
    --figure .\reports\figures\molecular_identity\11_standardization_status_by_partition.png
```

### Outputs

- `data/interim/M1_8c_standardized_observations.csv`: row-for-row, original M1.7 evidence plus identity decision and flags. **Keep out of public Git**.
- `reports/audits/M1_8c_standardization_summary.json`: input SHA256s, policy/version, status and partition counts, provisional EGFR/HER2 overlaps and limitations.
- `reports/audits/M1_8c_representation_collisions.csv`: representation-level collisions requiring interpretation (not automatically errors).
- `reports/figures/molecular_identity/11_standardization_status_by_partition.png`: visualization of the observation-level hold policy.
- `reports/audits/M1_8c_interpretation.md`: evidence-based interpretation and scientific caveats.

### Important limitations

- The source protein construct policy remains **provisional**, not sequence-verified wild-type.
- `RDKISO1` is a **structural representation ID**, not a confirmed assay-material ID. Its semantics can change if RDKit canonicalization or later curation policy changes.
- A known stereocenter with unspecified configuration is not a separate, guaranteed racemate or known enantiomer; flag and preserve this uncertainty.
- The same canonical structure can appear in multiple targets, variants, sources, and assays. No potency labels are aggregated here.
- To test generalization, maintain molecular/scaffold disjointness across train/validation/test; construct-shift and same-molecule within-assay contrasts are separate analyses.

**Important version note:** The example summary bundled here used RDKit 2025.09.4 for *stereochemistry flags* and the uploaded M1.8a canonical SMILES from RDKit 2026.09.1 for structure IDs; regenerate locally to keep the provenance consistent.

**Next milestone:** M1.8d — chemistry-specific collision review, then M1.9 observation-level assay disagreement and molecule–target label curation. Do not rush to average potency values.
