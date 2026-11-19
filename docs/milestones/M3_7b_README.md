# M3.7b — Fresh fit-only RF/GINE refit and 90% split-conformal intervals

## Prerequisites

- M3.7a conformal foundation is installed and tested.
- Corrected M2 RF Optuna SQLite studies exist in `reports/optuna/rf_fit_only_seed719.db`.
- Frozen M3.5 GINE five-fold reference report exists in `reports/milestones/M3_5_GNN_reference_5fold_cv.json`.
- MLflow HTTP server is available at `http://127.0.0.1:5000`.
- You are in the repository root.

## Scientific protocol

1. Recreate the frozen M2.1 Stage C split (fit=5,364; calibration=1,658; threshold=1,670); refuse changed dataset/assignment hashes.
2. Validate the two frozen Optuna studies and refit two distinct Morgan RF regressors on fit-role labeled molecules only.
3. Inspect the *already existing* GINE fit-only five-fold reference report, verify the original selected epochs 42, 45, 47, 48, and 76; select their median **47** as a predeclared final refit duration.
4. Initialize a fresh GINE and train 47 epochs on all fit-role molecular graphs. No early stopping or tuning using calibration labels.
5. Predict the entire calibration role with the frozen RF models and GINE. Derive one fixed 75/25 ensemble prediction per target.
6. Fit six separate **90% split-conformal** absolute-error quantiles, one per model (RF / GINE / RF75-GINE25) and target (EGFR / HER2).
7. Save model checkpoints, their SHA-256 digests, all calibration predictions (without measured labels), and calibrator metadata. Verify restored-model inference. Log the evidence to the existing MLflow experiment.

**No development validation, threshold tuning, or locked test scoring occurs in this script.** Calibration residuals are used strictly to determine the six interval widths. The script does **not** report calibration-set coverage as an external evaluation.

This Stage C calibration role belongs to the frozen training partition that participated in the earlier RF Optuna CV; therefore it is not independent of all historical model selection. Coverage under arbitrary new-scaffold distribution shift is not guaranteed. Treat intervals as estimated uncertainty and empirically audit coverage in a separately authorized development-validation step.

## Commands

```powershell
uv run --locked --group gnn pytest -v ml/tests/test_final_conformal_driver.py
uv run --locked --group gnn ruff check .
uv run --locked --group gnn pytest -q

uv run --locked --group gnn python scripts/fit_rf_gine_conformal.py `
  --device cuda `
  --mlflow-uri "http://127.0.0.1:5000"
```

Run once, unless a documented rerun is necessary (`--overwrite` intentionally required).

## Artifacts

- `artifacts/models/M3_7b_final_RF_EGFR.joblib`
- `artifacts/models/M3_7b_final_RF_HER2.joblib`
- `artifacts/models/M3_7b_final_GINE_fit_only.pt`
- `reports/milestones/M3_7b_calibration_predictions.csv`
- `reports/milestones/M3_7b_conformal_calibrators.json`

## Next milestone

M3.7c: use frozen checkpoints and calibrators to evaluate empirical 90% coverage, mean interval width, MAE, and difficult-scaffold groups on the outer **development validation** partition. Freeze model and reporting protocol before the one-time locked-test evaluation. The current milestone deliberately does not read the locked test labels.
