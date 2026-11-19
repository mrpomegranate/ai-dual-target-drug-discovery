# M3.7d — One-time locked-test evaluation protocol

## Purpose and precommitment

**Test candidates are fixed before reading any locked-test potency labels:**

- **RF:** M3.7b final EGFR and HER2 regressors, selected using M2.2 fit-only CV.
- **GINE:** M3.7b final GINE model, 3 message-passing layers, 128-dimensional hidden states, fixed 47-epoch refit.
- **Ensemble:** Fixed 75% RF / 25% GINE pIC50 prediction for each target.

The M3.7b six **frozen** conformal 90% marginal interval quantiles are applied unchanged, with one quantile per target × model. Test data are never used for refitting, model selection, weighting, temperature scaling, calibrating, or decision-threshold tuning. No GATv2 test comparison is included because the three-candidate protocol was previously specified.

**Earlier development-validation finding:** HER2 conformal undercoverage (RF 84.1%, GINE 81.6%, ensemble 83.4%) is documented; no post-hoc recalibration is permitted for this locked test.

## Installation

Extract archive into the repository root. Run:

```powershell
uv run --locked --group gnn pytest -v ml/tests/test_locked_test_io.py ml/tests/test_locked_test_driver.py
uv run --locked --group gnn ruff check .
uv run --locked --group gnn pytest -q
```

## Phase 1: preflight (safe; **no test labels are parsed or scored**)

```powershell
uv run --locked --group gnn python scripts/evaluate_rf_gine_locked_test.py `
  --preflight --device cuda --mlflow-uri "http://127.0.0.1:5000"
```

Preflight verifies hashes of frozen M2.1 data, M3.7b calibrators and all three model checkpoints; compares the M3.7c development-validation report against the calibrator report; regenerates the frozen Stage C assignment from **test-blind** `load_modeling_pipeline()`; and exercises checkpoint deserialization and a two-molecule fit-role inference smoke check. It also checks that the MLflow experiment exists, then writes `reports/milestones/M3_7d_frozen_test_protocol.json`. It prints **Protocol SHA-256**, which must be retained. Running preflight again without changing files is idempotent.

Inspect the frozen protocol before proceeding. The preflight file records the fixed model recipes, scoring definitions, code hashes, and artifact hashes. The original M2.1 loader remains completely test-blind.

## Phase 2: explicitly authorized ONE-TIME evaluation

Only execute once, after all artifact checks pass and you have committed to the final scientific protocol:

```powershell
uv run --locked --group gnn python scripts/evaluate_rf_gine_locked_test.py `
  --evaluate --device cuda `
  --confirm-unlock UNLOCK_M3_7D_TEST_ONCE `
  --expected-protocol-sha256 "<PASTE_SHA256_FROM_PREFLIGHT>" `
  --mlflow-uri "http://127.0.0.1:5000"
```

This requires the exact SHA from preflight. It refuses to run if any input, checkpoint, scorer code, or model recipe changes; if results already exist; or if the consumed-test marker is present. It opens the existing MLflow experiment before touching test labels.

**The one-time marker is created *before* accessing test labels:**

`reports/milestones/M3_7d_LOCKED_TEST_CONSUMED.json`

Even if the run errors afterward, do **not** delete the marker and rerun. Preserve the partial artifacts and failure log; only manually recover artifact logging or analyze already-produced predictions without rerunning model inference or changing the precommitted procedure.

## Final outputs

Under `reports/milestones/`, using `M3_7d_locked_test_...` prefixes:

- predictions CSV (1,087 molecules), including measured labels and all three models' predictions;
- accuracy, Spearman, marginal 90% conformal coverage, interval width CSV/JSON;
- dual-activity ranking AP on double-labeled molecules with strict experimental pIC50 >6.0 for both proteins;
- scaffold-level error, bias, and coverage CSV; groups smaller than 10 are retained but not ranked as reliable;
- potency-bin error and coverage CSV; these bins use the measured test answer and are **descriptive** only;
- coverage and MAE figures; M3.7d MLflow run with protocol SHA and metrics;
- the one-time consumed-test marker.

**Handle test-predictions CSVs as research data**: do not publish record-level measured potency labels to a public repository without confirming data rights.

## Scientific interpretation and limitations

- Marginal conformal coverage is not guaranteed under arbitrary scaffold shift; descriptive scaffold coverage is not a conditional coverage guarantee.
- Separate 90% EGFR and HER2 intervals do **not** promise 90% simultaneous dual-target coverage; reported paired joint coverage is descriptive.
- This is a *single frozen external-to-training split*, not a comprehensive performance guarantee for de novo generated chemistry or wet-lab inhibitory activity.
- Earlier RF model selection used full original training CV (including molecules later assigned to Stage C calibration). The calibration procedure therefore does not have a pristine model-selection-independent conformal guarantee.
- Any follow-up changes after inspecting test metrics must be labeled **post-test exploratory**, not a new independent final test.
