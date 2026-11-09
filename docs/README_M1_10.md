# M1.10 — Provisional EGFR/HER2 potency-label reconciliation

**Status:** Implemented and independently exercised on the earlier uploaded M1.8c data; run the script in the locked local repository to establish authoritative outputs. No model has been trained.

## Scientific motivation

BindingDB contains repeated exact IC50 values for the same provisional molecular structure and target. M1.9a found 3,093 repeated molecule–target pairs, including 354 with conflicting exact active/inactive classifications (IC50 < 1000 nM vs >= 1000 nM). **M1.10 adds a further guard:** exact IC50 measurements must also be checked against **determinate censored** observations for that same molecule and target. An exact `100 nM` observation (active) and a censored `>1000 nM` observation (inactive) represent contradictory evidence. Neither should be silently erased.

The output is **provisional label candidates** for later model development and sensitivity analysis—not experimentally verified target/assay labels or final train/test examples. A plain target name is not proof of WT EGFR/HER2, and a publication/patent ID is not an independently verified assay or laboratory.

## Policies (v1)

1. Use only M1.7 `primary_exact_candidate` observations with M1.8c `provisional_exact_structure` status to propose numerical regression labels. Keep mutant/engineered construct reserve and chemical-identity holds out.
2. Group by **`target + m18c_structure_id`**. Do not join different targets, different canonical-isomeric structures, salts, tautomers, or stereoisomers. A structure ID is an evidence-tracking identity, not guaranteed experimental material equivalence.
3. If exact observations cross the strict 1000 nM activity threshold, **withhold** that regression label.
4. Also **withhold** a label if any determinate M1.7 primary censored observation for that structure and target has an opposite activity classification.
5. If a pair has one exact measurement and passes the conflict checks, label it `provisional_single_exact` (unreplicated, not high confidence).
6. If repeated exact observations all agree within a **0.30 pIC50 range** (approximately twofold in IC50), and the conflict checks pass, label `provisional_tight_consensus`.
7. Otherwise hold the pair as moderate or major disagreement (>1.0 pIC50 units), keeping all original records.
8. For proposed labels, use the **median of per-provenance-group pIC50 medians**, so one source with duplicated/repeated rows does not directly dominate the center. This is a heuristic and does **not** establish source independence.
9. Produce two alternate sensitivity flags: a relaxed 0.50 pIC50 agreement range and a strict 0.30 policy that *ignores* censored contradictions (comparison only, **not** recommended as the main benchmark).
10. Retain an observation-level trace CSV. No observation is edited, dropped from the raw source, or given an imputed exact pIC50. No scaffold split or GNN training occurs here.

## Results from uploaded M1.8c data (not final local run)

| Decision | EGFR pairs | HER2 pairs | Total |
|---|---:|---:|---:|
| Provisional single exact | 8,025 | 2,971 | **10,996** |
| Provisional tight consensus | 990 | 235 | **1,225** |
| Hold exact activity contradiction | 335 | 19 | 354 |
| Hold censored activity contradiction | 279 | 6 | 285 |
| Hold moderate disagreement | 701 | 65 | 766 |
| Hold major disagreement | 530 | 67 | 597 |
| **Total** | **10,860** | **3,363** | **14,223** |

- 20,649 exact primary records, plus 4,844 determinate censored records checked for possible contradiction.
- **12,221** provisional regression labels; **2,002** held.
- **672** pairs had at least one censored observation; **333** pairs had some censored/classification conflict (some already failed exact consistency). The exclusive `hold_censored_activity_contradiction` decision applies to 285 pairs.
- If censored evidence were ignored, 12,438 labels would pass the exact-only policy: **217 additional questionable labels**.
- Relaxing the pIC50 repeat threshold from 0.30 to 0.50 yields 12,447 accepted labels, with the censored veto still active. This is a sensitivity scenario, not a quality improvement.
- **1,683** provisionally accepted pairs also have observations for an annotated construct with the same structure/target; challenge-set leakage rules remain mandatory.

### Why this is not the final curated dataset

Source verification found that plain target names can coexist with mutant/cell-based assays, but the required record-to-source-table matching is incomplete. Therefore *all* M1.10 accepted labels carry `NOT_sequence_or_assay_verified`. They must not be presented as verified wild-type biochemical potency. We may later exclude a further subset when assay/construct information is validated.

## Repository files

```
scripts/reconcile_potency_labels.py
ml/tests/test_reconcile_potency_labels.py
docs/README_M1_10.md
reports/audits/M1_10_reconciliation_summary.json
reports/figures/label_reconciliation/18_provisional_label_decisions.png
```

The three row-level outputs go under `data/interim/` and should remain gitignored:

```
data/interim/M1_10_provisional_regression_labels.csv
data/interim/M1_10_label_sensitivity.csv
data/interim/M1_10_observation_trace.csv
```

## Run locally (PowerShell)

```powershell
uv run --locked pytest -q
uv run --locked ruff check scripts/reconcile_potency_labels.py ml/tests/test_reconcile_potency_labels.py

uv run --locked python .\scripts\reconcile_potency_labels.py `
    --input .\data\interim\M1_8c_standardized_observations.csv `
    --m18d-summary .\reports\audits\M1_8d_collision_summary.json `
    --m19a-summary .\reports\audits\M1_9a_disagreement_summary.json `
    --labels .\data\interim\M1_10_provisional_regression_labels.csv `
    --sensitivity .\data\interim\M1_10_label_sensitivity.csv `
    --observation-trace .\data\interim\M1_10_observation_trace.csv `
    --summary .\reports\audits\M1_10_reconciliation_summary.json `
    --figure .\reports\figures\label_reconciliation\18_provisional_label_decisions.png
```

The script checks the full standardized input's SHA-256 against **both** M1.8d and M1.9a reports and checks exact count reconciliation against M1.9a. It fails on provenance mismatch instead of silently proceeding.

## After M1.10

1. Inspect the local JSON summary and verify the checksum and counts.
2. Implement M1.11 paired EGFR/HER2 dataset assembly with missing-label masks and separate censored/construct datasets.
3. Before training, audit assay-construct uncertainty and prevent identical structures, related InChIKey families and Bemis–Murcko scaffolds from crossing train/validation/test boundaries.
4. Evaluate source/provenance robustness and label-policy sensitivity separately from the primary scaffold-held-out RF vs GNN benchmark.

**Never infer assay equivalence or laboratory bias from the accepted-label flag alone.**
