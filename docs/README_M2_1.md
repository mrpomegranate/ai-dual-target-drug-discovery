# M2.1 — Frozen scaffold-disjoint EGFR/HER2 partitions

**Repository:** `ai-dual-target-drug-discovery`  
**Expected branch:** `m2/models` (based on `m1/data-audit` at `9ab0b81`)  
**Status:** New script and 12 **synthetic-only** tests created and run in the preparation environment. **Not yet executed on the private M1.11 dataset on your computer.**  
**Scientific scope:** Split the frozen, *provisional* M1.11 regression labels. Do **not** retrain M1 labels, validate proteins, or train a model in M2.1.

## What was confirmed from your committed M1 code

The `M1_11_multitask_molecules.csv` schema has `m18c_structure_id`, `canonical_isomeric_smiles`, target-specific `*_mask` / `*_pic50`, `n_available_regression_tasks`, `paired_activity_class`, `training_split`, and `scaffold_split` columns. In the source M1.11 assembly code, `training_split` and `scaffold_split` are **both `NOT_ASSIGNED`**. The versioned M1.11 JSON reports `train_val_test_splits_created: false`. `m1/data-audit` does not contain `scripts/split_scaffold_multitask.py` or a corresponding test.

## Decisions and limitations

1. **Molecular identity is preserved:** use `m18c_structure_id` and revalidate its `RDKISO1-` SHA-256 digest; retain original SMILES and all assay/label columns untouched.
2. **Bemis–Murcko scaffold:** extract the rings and connecting linkers with RDKit, ignoring chirality for grouping. Molecules with the same core share `scaffold_group_id` and **must be assigned together**.
3. **Acyclic exception:** molecules with no ring have an empty Murcko scaffold. To avoid grouping hundreds of unrelated acyclic molecules into a single artificial core, each gets `ACYCLIC-<structure_id>`. This is an **identity-level** split for acyclic structures, *not evidence of scaffold OOD performance*.
4. **Deterministic allocation:** group assignment uses seed `714`, target proportions 80% train / 10% validation / 10% test, and only group sizes plus **presence/absence of labels** (EGFR, HER2, paired). The algorithm does *not* use pIC50 magnitudes or active/inactive labels when assigning groups. Because all members of a core remain together, achieved ratios will not always equal 80/10/10 exactly.
5. **Source version gate:** script compares local RDKit version with `inputs.rdkit_version_provenance` in the authoritative M1.11 summary (`2026.09.1` in the supplied run). Fix environment drift rather than silently freezing a different partition.
6. **No leakage:** manifest tests assert disjoint molecular IDs, disjoint scaffold group IDs, full molecule coverage and label-mask consistency. The model input keeps `training_split` and `scaffold_split` as the **partition name**, with an explicit new `scaffold_group_id` column to avoid ambiguity.
7. **Scientific limits:** scaffold-disjoint is not necessarily chemically dissimilar; similar fingerprints may span different scaffold IDs. A later Morgan/Tanimoto similarity or cluster-held-out analysis is an optional robustness test. Most M1 labels are single-observation and sequence/assay-unverified. Keep annotated-construct data out of this split.

## Install the ZIP into your repo

Work from the repository root, on your existing `m2/models` branch. The ZIP contains only three files:

```
scripts/split_scaffold_multitask.py
ml/tests/test_split_scaffold_multitask.py
docs/README_M2_1.md
```

In PowerShell, replace `<downloaded ZIP path>` with the actual downloaded path:

```powershell
git status -sb
Expand-Archive -LiteralPath '<downloaded ZIP path>' -DestinationPath .
uv run --locked python -c "import rdkit; print(rdkit.__version__)"
uv run --locked pytest -q
uv run --locked ruff check scripts/split_scaffold_multitask.py ml/tests/test_split_scaffold_multitask.py
```

`Expand-Archive` intentionally does **not** modify `pyproject.toml`, `uv.lock`, or the private data. If you have different files at the three destination paths, review before replacing them.

## Execute the actual split locally

```powershell
uv run --locked python .\scripts\split_scaffold_multitask.py `
    --input .\data\interim\M1_11_multitask_molecules.csv `
    --m11-summary .\reports\audits\M1_11_assembly_summary.json `
    --dataset .\data\interim\M2_1_scaffold_split_molecules.csv `
    --manifest .\data\interim\M2_1_scaffold_split_manifest.csv `
    --summary .\reports\audits\M2_1_scaffold_split_summary.json `
    --figure .\reports\figures\splitting\20_scaffold_disjoint_split.png `
    --seed 714
```

The command refuses to overwrite existing output files; if intentionally doing a fresh experiment, write new versioned paths instead of erasing your frozen split.

**Outputs:**

- `data/interim/M2_1_scaffold_split_molecules.csv`: all M1 columns, plus assigned `training_split` / `scaffold_split` and explicit `scaffold_group_id`, `murcko_scaffold_smiles`, `scaffold_type`.
- `data/interim/M2_1_scaffold_split_manifest.csv`: compact assignment mapping with molecule IDs, scaffold ID/type and masks.
- `reports/audits/M2_1_scaffold_split_summary.json`: actual partition counts, group counts, check results, input/output checksums, seed, RDKit version and policies.
- `reports/figures/splitting/20_scaffold_disjoint_split.png`: counts of molecules, EGFR labels, HER2 labels, and paired labels by partition.

## Local validation and handoff

```powershell
Get-Content .\reports\audits\M2_1_scaffold_split_summary.json
Get-FileHash .\data\interim\M2_1_scaffold_split_manifest.csv -Algorithm SHA256
Import-Csv .\data\interim\M2_1_scaffold_split_manifest.csv |
    Group-Object training_split | Select-Object Name,Count

git status -sb
```

**Share:** test / Ruff output, local RDKit version, and `M2_1_scaffold_split_summary.json`. You do not need to upload private BindingDB-derived molecular records.

**M2.1 acceptance:** no overlapping scaffold groups/molecule IDs, all 10,866 original structures and 9,015/3,206 labels represented unchanged, no missing label converted into a numerical value, viable per-target coverage in all three partitions, and manifest SHA-256 recorded. Once frozen, the test split is never used to select RF/GNN hyperparameters; begin M2.2 RF training using only train/validation partitions.
