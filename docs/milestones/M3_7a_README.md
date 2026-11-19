# M3.7a — Fit-only scaffold difficulty audit + split-conformal primitives

Status: **development tools implemented**. Calibration of the final RF, GINE and
75%-RF/25%-GINE ensemble **has NOT yet been performed**. The existing OOF
predictions must not be used to calculate final conformal radii.

## Rationale

A pooled MAE hides chemically heterogeneous errors. This step characterizes
frozen fit-only scaffold-held-out errors without accessing reserved labels,
then creates a strictly bounded conformal-calibration API for the next
fresh-model refit step.

The top-scaffold rankings are **exploratory development diagnostics**.
A scaffold with `n=1` can have a huge error by chance; the default ranking
requires **at least ten accepted labels per target and scaffold**.
The full per-scaffold report retains all groups so sample support remains
visible. Rankings use the RF error as a common baseline; do not interpret
these selected worst groups as unbiased prevalence estimates.

## Install and run

From the repository root after extracting this archive:

```powershell
uv run --locked --group gnn pytest -v ml/tests/test_conformal_regression.py ml/tests/test_scaffold_diagnostics.py ml/tests/test_scaffold_rendering.py
uv run --locked --group gnn ruff check .
uv run --locked --group gnn pytest -q

uv run --locked --group gnn python scripts/report_scaffold_difficulty.py `
  --oof reports/milestones/M3_6d_three_model_and_ensemble_fit_only_OOF.csv `
  --mlflow-uri "http://127.0.0.1:5000"
```

Required input: `reports/milestones/M3_6d_three_model_and_ensemble_fit_only_OOF.csv`
from the previous consolidated three-model OOF experiment. If it exists only
in your Downloads folder, copy it to the indicated reports location; do not
use a test prediction CSV as a substitute.

Outputs under `reports/milestones/`:

- `M3_7a_fit_only_scaffold_difficulty_all.csv` — all groups, target-specific
  support, MAE/median absolute error/signed bias for each model
- `M3_7a_fit_only_scaffold_difficulty_top.csv` — top 12 difficult RF scaffolds
  separately for EGFR and HER2 (minimum ten accepted observations)
- `M3_7a_fit_only_scaffold_difficulty_egfr.png` and `_her2.png`
- `M3_7a_fit_only_scaffold_difficulty.json` — provenance and counts

The default ranking is by RF MAE for a common comparison. To rank by GINE
errors instead, pass `--rank-by gine_mae`; supported alternatives are
`gatv2_mae` and `rf75_gine25_mae`. Existing outputs require `--overwrite`.

The report script logs the report and plots to your existing MLflow experiment.
It refuses to overwrite prior report files unless explicitly given `--overwrite`.

## To show molecular cores, not just hashes

The original scaffold IDs are hashes, not human-readable structures. For
figure-ready chemical examples, run:

```powershell
uv run --locked --group gnn python scripts/render_scaffold_difficulty.py
```

This uses the existing frozen dataset/manifest and constructs the audited
seed-719 Stage C split. **Only `split.fit` structures** are selected for
illustration; test-label loading remains blocked by the existing modeling
loader. A representative RDKit Bemis–Murcko core is drawn for each group.
Note that a representative core depicts one common motif and does not prove
all molecules in the group have identical atom/bond patterns. Acyclic
identity-groups are shown using a representative full molecule.

Output: `M3_7a_fit_only_scaffold_structures.csv` and two labeled molecule-grid
PNGs for EGFR/HER2.

## Conformal API ready for the following step

`dual_target.calibration.conformal_regression` exposes
`fit_split_conformal`, `SplitConformalCalibrator.intervals`, and
`summarize_interval_coverage`.

**Critical workflow:**

1. Freeze final RF configs, GINE configuration, **final training-epoch count**,
   ensemble weight (75/25), random seed and dataset/split hashes.
2. Train **fresh final models exclusively on the 5,364 `split.fit` molecules**.
3. Predict **all 1,658 `split.calibration` molecule IDs**, for each target and
   each of RF, GINE and the 75/25 ensemble. Do not use calibration errors to
   optimize the architecture, model, weights, or epochs.
4. Call `fit_split_conformal(split, predictions, target="egfr", model_name="rf")`
   and repeat separately for both targets and all model candidates. The API
   checks the exact calibration-role ID coverage, frozen Stage C hash,
   complete assignment integrity, and scaffold/identity disjointness.
5. Produce interval bounds and evaluate empirical coverage / width on the
   independent development-validation partition, and **only later** on the
   locked test in the one-time final evaluator. Group coverage and width by
   scaffold with sample-size thresholds.

This uses the conservative order statistic
`q = sorted(abs(y - yhat))[ceil((n+1)*(1-alpha)) - 1]`. For a given alpha,
when the required rank exceeds the number of accepted calibration labels,
the API fails instead of silently emitting a finite interval without support.
Masked target labels are never treated as zero or inactive.

### Coverage qualifications

- Split-conformal 90% is **marginal** coverage under exchangeability,
  not a per-molecule or per-scaffold guarantee.
- Fold-level scaffold disjointness does not guarantee exchangeability
  across a new chemical-series distribution shift.
- Some Stage C molecules were involved in earlier exploratory model
  development before the strict fit-only protocol was adopted. Therefore
  do not assert an unconditional formal coverage guarantee for this
  historical adaptive workflow.
- EGFR and HER2 marginal 90% intervals do not together guarantee 90%
  simultaneous coverage. Joint dual-target confidence needs separate
  handling (for example, Bonferroni-adjusted 95% per target) and validation.
- Intervals quantify uncertainty; **they do not correct point-prediction MAE**.

No calibration, threshold, development-validation, or locked-test labels
are accessed by the M3.7a scaffold-report script.
