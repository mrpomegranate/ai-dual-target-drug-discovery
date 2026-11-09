# M1.7 — Provisional construct inclusion policy

**Status:** Implemented against the M1.4 observation export; final molecule-level curation and validation are not complete.

## Research purpose

A molecule-only Random Forest (Morgan fingerprints) and molecular GNN receive a **ligand molecular structure**, plus an EGFR/HER2 task index; they do not receive the tested **protein sequence or construct**. Consequently, IC50 measurements against reference-like and explicitly altered EGFR/HER2 proteins must not be silently merged. A target name without an annotation is **not** confirmation of wild-type status.

M1.7 translates the M1.6 target-name classification into conservative, **provisional** dataset partitions. It does not establish an independent test set, perform a scaffold split, standardize ligands, combine assays, or train models.

## Policy (in priority order)

| Input observation | M1.7 partition | Notes |
|---|---|---|
| Recognized name annotation: substitution, insertion/deletion, mixed construct | `secondary_construct_reserve` | Every such observation stays reserved; `m17_assay_role` distinguishes exact, classifiable-censored and not-ready |
| Unrecognized non-base target name | `manual_construct_review` | Requires human inspection; not automatically assumed mutant |
| Plain target name with missing/nonhuman organism metadata | `primary_hold_organism` | Do not impute organism based only on UniProt accession |
| Plain target name, explicit human, missing SMILES | `primary_hold_smiles` | No ligand input available |
| Plain target name, explicit human, SMILES present, consistent exact positive IC50/pIC50 | `primary_exact_candidate` | Preliminary potency regression pool; not yet standardized or assay-reconciled |
| Plain target name, explicit human, SMILES present, censored IC50 yielding definitive active/inactive at 1000 nM | `primary_classification_candidate` | Binary-only; **no exact pIC50** |
| Remaining plain-name measurements | `primary_hold_measurement` | E.g. invalid or indeterminate censored values |

The M1.7 script normalizes raw organism strings `Human` and `Homo sapiens`, and **reports discrepancies** with the older `organism_flag`. It never changes the M1.4 export.

## Files

- `scripts/apply_construct_inclusion_policy.py` — policy/validation, report and two figures
- `ml/tests/test_apply_construct_inclusion_policy.py` — 20 synthetic unit/integration tests
- `data/interim/M1_7_observation_decisions.csv` — **generated locally**, observation-level data, **do not commit**
- `reports/audits/M1_7_construct_policy_summary.json` — committed audit metrics with input SHA-256
- `reports/audits/M1_7_construct_policy_notes.md` — interpretation
- `reports/figures/source_bias/09_provisional_inclusion_decisions.png` — counts by target and decision
- `reports/figures/source_bias/10_source_composition_after_policy.png` — source composition shift

**Dependency:** Keep M1.6 `scripts/audit_construct_provenance.py`; M1.7 imports its tested target-name classifier.

## Reproduce (Windows PowerShell, from repository root)

Extract the supplied ZIP **into the project root** (merge `scripts/`, `ml/tests/`, `docs/`, and `reports/`). No additional package installation should be required if the M1.6 and M1.5 dependencies are installed.

```powershell
uv run --locked pytest -q
uv run --locked ruff check scripts/apply_construct_inclusion_policy.py ml/tests/test_apply_construct_inclusion_policy.py
```

```powershell
uv run --locked python .\scripts\apply_construct_inclusion_policy.py `
    --input .\data\interim\bindingdb_egfr_her2_observations.csv `
    --decisions .\data\interim\M1_7_observation_decisions.csv `
    --summary .\reports\audits\M1_7_construct_policy_summary.json `
    --figure-dir .\reports\figures\source_bias
```

Review counts:

```powershell
Get-Content .\reports\audits\M1_7_construct_policy_summary.json -Raw
```

## Expected reconciliation from the supplied 35,210-row snapshot

| Partition | EGFR | HER2 | Total |
|---|---:|---:|---:|
| Primary exact regression candidates | 16,517 | 4,148 | 20,665 |
| Primary classifiable censored candidates | 4,267 | 578 | 4,845 |
| Primary measurement holds | 213 | 10 | 223 |
| Annotated-construct reserve, **all assay types** | 8,218 | 1,259 | 9,477 |
| **All observations** | **29,215** | **5,995** | **35,210** |

Within the annotated reserve, just **6,261 EGFR and 793 HER2** records passed the *initial exact IC50, structure and human metadata* checks. These are not yet finalized challenge test examples.

The supplied snapshot predates the corrected M1.4 `organism_flag`: it has **3,409 flag discrepancies** against raw human metadata. Rerunning on the corrected local M1.4 export is authoritative; the discrepancy count should then be zero, while the scientific partition counts are expected to remain the same.

## What remains before the construct test is valid

1. Verify actual protein sequence and experimental context where possible, and distinguish mutation types.
2. Standardize molecular identities, investigate reported InChIKey inconsistencies, reconcile repeat IC50 measurements and sources.
3. For the **primary** benchmark, split molecule groups by Bemis–Murcko scaffold. Assess both targets without scaffold leakage.
4. For a **chemistry-generalization construct challenge**, reserve annotated-construct scaffolds disjoint from **all training scaffolds**. Report actual test set sizes and source imbalance.
5. Separately, conduct a **matched-molecule reference-versus-mutant assay comparison**, which intentionally uses the same molecule in two constructs. It is **not** a scaffold-held-out generalization test, and a molecule-only model cannot directly predict the mutation effect.
6. Source category is not an individual lab; assess source/patent/paper effects as confounders and consider study-held-out robustness when sample sizes allow.

## Important interpretation

An error increase on the annotated-construct challenge could reflect a change of biological target, source/protocol shifts, different molecular chemistry, or a combination. Because the current model lacks protein construct input, the result would demonstrate **limits of transfer**, not automatically a flaw in the GNN architecture or evidence that particular laboratories are biased.
