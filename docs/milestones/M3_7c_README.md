# M3.7c — Frozen development-validation coverage and scaffold diagnostics

## Scope

This step **reads** the already-fitted M3.7b EGFR RF, HER2 RF, and GINE checkpoints, plus their six **fixed** 90% absolute-residual split-conformal calibrators. It derives the **fixed** 75% RF / 25% GINE ensemble without refitting or selecting weights. It scores the **frozen M2.1 development-validation** partition (1,087 molecules) only. The locked test set is not loaded or scored.

### Inputs

- `data/interim/M2_1_scaffold_split_molecules.csv`
- `data/interim/M2_1_scaffold_split_manifest.csv`
- `reports/audits/M2_1_scaffold_split_summary.json`
- `reports/milestones/M3_7b_conformal_calibrators.json`
- `artifacts/models/M3_7b_final_RF_EGFR.joblib`
- `artifacts/models/M3_7b_final_RF_HER2.joblib`
- `artifacts/models/M3_7b_final_GINE_fit_only.pt`

Each checkpoint's SHA-256 must match the M3.7b report. The original dataset, outer manifest, Stage C assignment and graph schema are verified. The training/calibration/threshold roles must be disjoint from development validation by molecule identity and scaffold. All data access is through `load_modeling_pipeline()`, which excludes test labels.

### Installation

Extract the ZIP at the repository root. Then execute:

```powershell
uv run --locked --group gnn pytest -v ml/tests/test_final_validation.py ml/tests/test_dev_validation_driver.py
uv run --locked --group gnn ruff check .
uv run --locked --group gnn pytest -q
```

### Execute

With your existing MLflow server running:

```powershell
uv run --locked --group gnn python scripts/evaluate_rf_gine_dev_validation.py `
  --device cuda `
  --mlflow-uri "http://127.0.0.1:5000"
```

This job refuses to overwrite prior M3.7c files without `--overwrite`.

### Outputs (under `reports/milestones/`)

- `M3_7c_dev_validation_predictions.csv`: frozen measured masks, values and three models' point predictions on 1,087 molecules.
- `M3_7c_dev_validation_metrics.csv`: six rows, one target × model, with MAE, Spearman, marginal coverage, and mean interval width.
- `M3_7c_dev_validation_metrics.json`: model metrics, dual-activity AP, descriptive paired joint coverage, scaffold statistics, provenance and MLflow run ID.
- `M3_7c_dev_validation_scaffold_coverage.csv`: target × model × scaffold performance, with explicit minimum-N flag.
- `M3_7c_dev_validation_potency_coverage.csv`: subgroup MAE, prediction bias and interval coverage for measured pIC50 `<6`, `6–<7`, `7–<8` and `>=8`.
- `M3_7c_dev_validation_coverage.png`: six empirical marginal coverage estimates against nominal 90%.

### Interpretation / limitations

- **Empirical 90% coverage** is a measured *marginal* frequency across held-out molecules, not a 90% posterior probability assigned to each individual prediction.
- **Subgroup scaffold coverage is descriptive**, not a conditional-coverage guarantee. Small scaffolds are included in the raw CSV but flagged as not reliable for rankings when fewer than 10 measured molecules occur.
- A fixed symmetric conformal q yields a constant width for a given target and model; width may be uninformative when plotted within a single model subgroup.
- The two target-specific 90% intervals are not jointly guaranteed to cover both targets at 90%. Simultaneous coverage for paired molecules is reported for exploration only.
- The frozen RF hyperparameters had previously been selected using fit-only training CV, which limits pristine post-selection calibration claims. Arbitrary scaffold distribution shift can also compromise marginal coverage.
- Do not reoptimize hyperparameters, ensemble weights or conformal quantiles using these validation labels and then describe the same validation results as independent.
- **No locked-test labels are used.** A separate, explicitly authorized one-time final test evaluator will be required after the protocol is frozen.
