# M1.11 — Masked EGFR/HER2 provisional paired dataset assembly

**Status:** Implementation tested on the available M1.10/M1.8c files. Run on your local locked repository to establish the authoritative results. **No ML model trained, no train/validation/test split and no confirmed wild-type assay labels.**

## Purpose

M1.10 made a **target-specific** decision for each of 14,223 EGFR or HER2 molecule–target pairs and accepted 12,221 provisional pIC50 labels. A multi-task molecular GNN, however, requires **one row per molecule** with a possibly missing label for each of two targets. M1.11 produces this reversible wide-form table without inventing missing measurements or bringing the construct-shift reserve into the primary candidate table.

## Policy, definitions, and safeguards

1. **Identity key:** `m18c_structure_id`, a SHA-256 digest of the full isomeric-canonical RDKit SMILES under `rdkit-isomeric-smiles-policy-v1`. The script independently recomputes each digest. Identity remains *provisional*: no salts, tautomers, stereoisomers or compounds are silently merged.
2. **Available target-specific labels:** Only M1.10 statuses `provisional_single_exact` and `provisional_tight_consensus` produce numerical labels.
3. **Missing labels:** For each target, `*_mask = 1` means a provisional numeric pIC50 is present. `*_mask = 0` means the pIC50 cell is **blank**, *not zero*. Mask=0 includes both missing exact evidence and withheld/conflicted evidence; distinguish these cases using `*_status`.
4. **Activity threshold:** Exact pIC50 > 6 means IC50 <1000 nM (active). Equal to 6 is **inactive** under the *strict* threshold. The `paired_activity_class` is available only if **both masks are 1**.
5. **Cross-target overlap:** EGFR and HER2 labels can appear on one structure row, but each target retains its own provenance counts and status. **One withheld target must not exclude the other accepted target.**
6. **Observation lineage:** Independently verify each M1.10 row's exact source observation IDs against M1.8c, including target, structure ID, canonical SMILES, min/max pIC50 and counts. Verify 20,649 source exact observations are represented exactly once by the M1.10 candidate entries.
7. **Construct-shift separation:** Preserve **all** M1.7 `secondary_construct_reserve` rows in a separate ignored CSV, even the 56 without approved chemical identities. These records are *not* primary training labels, even when they share the same structure ID as an accepted reference-like target observation.
8. **Audit trail:** A separate pair disposition table preserves **all 14,223** M1.10 decisions, including 2,002 held pairs and 1,695 held-only structures that do not enter model rows.
9. **Provenance:** SHA-256 of the M1.8c standardized source file must match the authoritative local M1.10 JSON. Both the label CSV and summary JSON also receive hashes in the M1.11 report.
10. **No leakage claims:** This stage does not assign scaffold groups or train/validation/test partitions. Identical structures and Bemis–Murcko scaffolds still need molecule-/scaffold-disjoint splits in the next milestone. Construct-shift stress tests need separate overlap controls.

## Preview results (from source files in the conversation)

| Category | Unique molecular structures |
|---|---:|
| Both EGFR and HER2 provisional labels | **1,355** |
| Only EGFR label accepted | **7,660** |
| Only HER2 label accepted | **1,851** |
| **Total unique candidate model rows** | **10,866** |
| Held-only identities, excluded from candidate rows | 1,695 |

- **12,221 provisional labels** overall: EGFR 9,015; HER2 3,206.
- Accepted label support: 10,996 single exact observations; 1,225 tight-consensus target labels.
- Two-target accepted subset activity categories (strict IC50 cutoff): 863 provisional dual-active; 280 dual-inactive; 142 EGFR-only active; 70 HER2-only active. **These are not experimental validation claims or ML predictions.**
- Full pair disposition audit retains **14,223** target-specific entries, of which 2,002 are withheld.
- Separate annotated construct reserve contains **9,477 original observation rows**, including 9,421 with approved structure IDs and 56 held structures.
- **1,683 accepted target-specific pairs** have corresponding annotated-construct observations, making construct-related leakage an especially important future test.

## Output locations (data outputs are private and Git-ignored)

```text
scripts/assemble_multitask_dataset.py
ml/tests/test_assemble_multitask_dataset.py
docs/README_M1_11.md
reports/audits/M1_11_assembly_summary.json
reports/figures/dataset_assembly/19_multitask_label_coverage.png

data/interim/M1_11_multitask_molecules.csv
data/interim/M1_11_all_pair_dispositions.csv
data/interim/M1_11_construct_shift_reserve.csv
```

**Do not commit the three `data/interim/` CSVs to the public repository.** The downloadable ZIP does *not* contain original BindingDB observation records.

## 1 — Extract and validate in PowerShell (repository root)

```powershell
uv run --locked pytest -q
uv run --locked ruff check scripts/assemble_multitask_dataset.py ml/tests/test_assemble_multitask_dataset.py
```

The 17 new tests passed in the preparation environment; your previous local suite ran 198 tests, so **about 215** should run afterward if no other tests have changed.

## 2 — Assemble the private, masked dataset

```powershell
uv run --locked python .\scripts\assemble_multitask_dataset.py `
    --labels .\data\interim\M1_10_provisional_regression_labels.csv `
    --m10-summary .\reports\audits\M1_10_reconciliation_summary.json `
    --standardized .\data\interim\M1_8c_standardized_observations.csv `
    --dataset .\data\interim\M1_11_multitask_molecules.csv `
    --pair-audit .\data\interim\M1_11_all_pair_dispositions.csv `
    --construct-reserve .\data\interim\M1_11_construct_shift_reserve.csv `
    --summary .\reports\audits\M1_11_assembly_summary.json `
    --figure .\reports\figures\dataset_assembly\19_multitask_label_coverage.png
```

### Example row (illustration ONLY, no real compound)

| structure_id | egfr_pic50 | egfr_mask | her2_pic50 | her2_mask | paired_activity_class |
|---|---:|---:|---:|---:|---|
| molecule-A | 7.30 | 1 | 6.80 | 1 | dual_active |
| molecule-B | 6.20 | 1 | *(blank)* | 0 | not_evaluable_missing_task_label |
| molecule-C | *(blank)* | 0 | 7.10 | 1 | not_evaluable_missing_task_label |

`egfr_status` or `her2_status` distinguishes *no primary exact evidence* from *held due to inconsistency*. All provisional labels carry the caution `NOT_sequence_or_assay_verified`.

### What to send back

Please share test/Ruff/assembly terminal output, and upload:

`reports/audits/M1_11_assembly_summary.json`

**Do not need to upload the three private row-level CSVs** unless we need a specific diagnostic.

## After M1.11 — scaffold/identity leakage-safe split

1. Compute Bemis–Murcko scaffolds using a recorded RDKit version. Empty scaffolds (acyclic molecules) need an explicit documented grouping decision.
2. Partition by scaffold while keeping all instances of a given M1.8c structure ID in one partition; preserve target label masks and ideally balance the 1,355 paired-labeled molecules.
3. Freeze train/validation/test assignment and save a manifest and label-policy version; no tuning on the test set.
4. Inspect stereoisomer/structure-collision and same-compound-across-reserve risks; annotated constructs are a **separate stress test**, not a primary test-set label source.
5. Train fingerprint + Random Forest and masked multi-task GNN on identical assigned train/validation/test populations. Analyze performance by support, source and sensitivity policies.

> **Scientific limitation:** A successful assembly is a computational QA milestone, not proof that included records correspond to wild-type EGFR/HER2 kinase inhibition. Source-level assay/construct verification remains incomplete and is documented as residual uncertainty.
