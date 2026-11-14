"""Stage D.3: fit-only Optuna EGFR/HER2 split-conformal RF intervals.

Reconstruct the frozen seed-719 scaffold roles. Read and verify the NEW
reference-first Optuna studies tuned using only the 5,364 fitting molecules
(five internal scaffold-disjoint folds), then train two 200-tree regressors
on the fitting labels. Compute fixed 90% conformal residual intervals using
only the disjoint calibration labels; reserve the threshold subset.

Report empirical coverage/MAE on development validation; never access locked
test labels. Neither scaffold shift nor label-informed choice of split seed
719 guarantees formal distribution-free coverage on unseen scaffolds.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from calibrate_dual_rf import (
    DEFAULT_AUDIT,
    DEFAULT_DATASET,
    DEFAULT_MANIFEST,
    DEFAULT_SPLIT_MANIFEST,
    DEFAULT_SUMMARY,
    FROZEN_ASSIGNMENT_SHA256,
    STAGE_B_RUN_ID,
    STAGE_C_AUDIT_RUN_ID,
    STAGE_C_SELECTION_RUN_ID,
    _file_sha256,
    configure_mlflow,
    git_state,
    verify_frozen_assignment,
)
from dual_target.calibration.conformal import (
    PotencyIntervalScores,
    SplitConformalRegressor,
    conformal_absolute_quantile,
    evaluate_conformal_development_validation,
    fit_split_conformal_regression,
)
from dual_target.calibration.scaffold_calibration import make_scaffold_calibration_split
from dual_target.data.pipeline import (
    build_fit_only_tuning_pipeline,
    load_modeling_pipeline,
)
from dual_target.features.morgan_pipeline import build_morgan_feature_bank
from evaluate_optimized_rf import read_selections

TARGETS = ("egfr", "her2")
DEFAULT_FIT_ONLY_OPTUNA_DB = Path("reports/optuna/rf_fit_only_seed719.db")
FIT_ONLY_OPTUNA_PARENT_RUN_ID = "2c748d4495ec47a4924d66043349709b"
STAGE_D1_RUN_ID = "7f604eb2e3514ed9802d6d22fa304fb2"
EXPECTED_FIT_ONLY_SELECTIONS = {
    "egfr": {
        "best_trial": 13,
        "max_depth": None,
        "min_samples_leaf": 2,
        "min_samples_split": 3,
        "max_features": 0.5,
    },
    "her2": {
        "best_trial": 0,
        "max_depth": None,
        "min_samples_leaf": 1,
        "min_samples_split": 2,
        "max_features": "sqrt",
    },
}
EXPECTED_LABEL_COUNTS = {
    "egfr": {"fit": 4465, "calibration": 1382, "threshold": 1364, "validation": 902},
    "her2": {"fit": 1596, "calibration": 499, "threshold": 470, "validation": 320},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--split-audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--study-db", type=Path, default=DEFAULT_FIT_ONLY_OPTUNA_DB)
    parser.add_argument("--study-prefix", default="M2_2_fit_only_reffirst_200trees")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-completed-trials", type=int, default=20)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.10,
                        help="Miscoverage level; default 0.10 for 90%% intervals")
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--experiment", default="EGFR_HER2_M2_Scaffold_Benchmark")
    parser.add_argument("--run-name", default="M2_2_StageD3_FitOnly_Potency_RF_Conformal")
    args = parser.parse_args()
    if args.trees != 200:
        parser.error("--trees must be 200 to reproduce the selected Optuna studies")
    if args.folds != 5 or args.min_completed_trials < 20:
        parser.error("--folds must be 5 and --min-completed-trials must be >= 20")
    if args.n_jobs == 0:
        parser.error("--n-jobs cannot be zero")
    if not math.isfinite(args.alpha) or not 0.0 < args.alpha < 1.0:
        parser.error("--alpha must be a finite number strictly between 0 and 1")
    return args


def _verify_target_counts(split, pipeline) -> None:
    for target in TARGETS:
        expected = EXPECTED_LABEL_COUNTS[target]
        for role in ("fit", "calibration", "threshold"):
            observed = getattr(getattr(split, f"{role}_audit"), f"{target}_labels")
            if observed != expected[role]:
                raise ValueError(f"Frozen {target} {role} label count changed")
        validation = sum(
            getattr(r, f"{target}_mask")
            for r in pipeline.data.validation
        )
        if validation != expected["validation"]:
            raise ValueError(f"Frozen {target} development-validation labels changed")


def _verify_fit_only_optuna_selections(selections: dict, provenance: dict) -> None:
    """Ensure the new 20-trial fit-only studies are the expected frozen winners."""
    for target, expected in EXPECTED_FIT_ONLY_SELECTIONS.items():
        meta = provenance[target]
        config = selections[target]
        if meta["best_trial"] != expected["best_trial"]:
            raise ValueError(f"{target.upper()} Optuna winner changed from frozen run")
        if meta["completed_trials"] != 20:
            raise ValueError(f"{target.upper()} study must have exactly 20 completed trials")
        if any(
            getattr(config, name) != value
            for name, value in expected.items()
            if name != "best_trial"
        ):
            raise ValueError(f"{target.upper()} selected RF settings changed")
        if config.n_estimators != 200 or config.seed != 714:
            raise ValueError(f"{target.upper()} unexpected tree count or model seed")


def _log_role_identities(pipeline, split) -> None:
    """Record the actual fit/calibration/validation roles without potency labels."""
    for role, records in (
        ("fit", split.fit),
        ("calibration", split.calibration),
        ("validation", pipeline.data.validation),
    ):
        frame = pd.DataFrame({
            "molecule_id": [row.structure_id for row in records],
            "scaffold_id": [row.scaffold_group_id for row in records],
        })
        mlflow.log_input(
            mlflow.data.from_pandas(frame, name=f"M2_2_stage_d3_{role}_identities"),
            context="training" if role == "fit" else role,
        )


def _interval_frame(scores: PotencyIntervalScores) -> pd.DataFrame:
    return pd.DataFrame({
        "molecule_id": scores.molecule_ids,
        "scaffold_id": scores.scaffold_ids,
        "observed_pic50": scores.observed_pic50,
        "predicted_pic50": scores.predicted_pic50,
        "lower_pic50": scores.lower_pic50,
        "upper_pic50": scores.upper_pic50,
        "absolute_residual": np.abs(scores.observed_pic50 - scores.predicted_pic50),
        "covered": scores.covered,
    })


def _metric_row(target: str, role: str, scores: PotencyIntervalScores) -> dict:
    return {
        "target": target,
        "split": role,
        **asdict(scores.metrics),
    }


def _plot_parity_with_intervals(target: str, scores: PotencyIntervalScores) -> None:
    """All validation points, with intervals on a deterministic small sample."""
    predicted = scores.predicted_pic50
    observed = scores.observed_pic50
    fig, ax = plt.subplots(figsize=(7.3, 6.3), dpi=180)
    try:
        ax.scatter(predicted[scores.covered], observed[scores.covered],
                   s=18, alpha=0.38, label="Observed within interval", color="tab:blue")
        ax.scatter(predicted[~scores.covered], observed[~scores.covered],
                   s=22, alpha=0.8, label="Observed outside interval", color="tab:orange")
        indices = np.argsort(predicted)[
            np.linspace(0, len(predicted) - 1, min(45, len(predicted)), dtype=int)
        ]
        ax.vlines(predicted[indices], scores.lower_pic50[indices],
                  scores.upper_pic50[indices], alpha=0.28,
                  linewidth=0.9, color="tab:green",
                  label=f"{len(indices)} illustrative prediction intervals")
        bounds = np.concatenate((observed, scores.lower_pic50, scores.upper_pic50))
        lower, upper = float(np.min(bounds) - 0.2), float(np.max(bounds) + 0.2)
        ax.plot([lower, upper], [lower, upper], "--", color="gray", linewidth=1,
                label="Ideal parity")
        ax.set(xlabel="Predicted pIC50", ylabel="Observed pIC50", xlim=(lower, upper),
               ylim=(lower, upper), title=f"{target.upper()} — pIC50 parity and intervals")
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        mlflow.log_figure(fig, "plots/validation_parity_intervals.png")
    finally:
        plt.close(fig)


def _plot_residuals(target: str, model: SplitConformalRegressor,
                    validation: PotencyIntervalScores) -> None:
    cal = np.abs(
        model.calibration_scores.observed_pic50 - model.calibration_scores.predicted_pic50
    )
    val = np.abs(validation.observed_pic50 - validation.predicted_pic50)
    fig, ax = plt.subplots(figsize=(7, 4.6), dpi=180)
    try:
        limit = float(max(cal.max(), val.max(), model.half_width))
        bins = np.linspace(0.0, max(0.1, limit * 1.02), 28)
        ax.hist(cal, bins=bins, density=True, histtype="step", linewidth=2,
                label=f"Calibration residuals (n={len(cal)})")
        ax.hist(val, bins=bins, density=True, histtype="step", linewidth=2,
                label=f"Validation residuals (n={len(val)})")
        ax.axvline(model.half_width, color="tab:red", linestyle="--",
                   label=f"Conformal half-width = {model.half_width:.3f}")
        ax.set(xlabel="Absolute pIC50 error", ylabel="Density",
               title=f"{target.upper()} — held-out residual distributions")
        ax.legend(fontsize=8)
        fig.tight_layout()
        mlflow.log_figure(fig, "plots/calibration_validation_residuals.png")
    finally:
        plt.close(fig)


def _confidence_sensitivity(
    model: SplitConformalRegressor, validation: PotencyIntervalScores
) -> pd.DataFrame:
    """Diagnostic-only curve; NEVER choose target coverage from validation."""
    calibration_residual = np.abs(
        model.calibration_scores.observed_pic50 - model.calibration_scores.predicted_pic50
    )
    validation_residual = np.abs(
        validation.observed_pic50 - validation.predicted_pic50
    )
    coverage_grid = tuple(sorted({0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
                                  round(1.0 - model.alpha, 8)}))
    rows: list[dict] = []
    for nominal in coverage_grid:
        try:
            use_alpha = model.alpha if abs(nominal - (1.0 - model.alpha)) < 1e-9 else 1.0 - nominal
            radius, order = conformal_absolute_quantile(
                calibration_residual, alpha=use_alpha
            )
        except ValueError:
            # Extreme nominal levels can require more residuals to form a finite radius.
            continue
        rows.append({
            "nominal_coverage": nominal,
            "empirical_validation_coverage": float(np.mean(validation_residual <= radius)),
            "interval_width_pic50": 2.0 * radius,
            "half_width_pic50": radius,
            "order_statistic": order,
            "n_validation": len(validation_residual),
        })
    return pd.DataFrame(rows)


def _plot_coverage_sensitivity(target: str, frame: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 5), dpi=180)
    try:
        ax.plot(frame["nominal_coverage"], frame["empirical_validation_coverage"],
                marker="o", linewidth=1.8, label="Observed validation coverage")
        ax.plot([0.65, 1.0], [0.65, 1.0], linestyle="--", color="gray",
                label="Nominal = empirical")
        ax.set(xlim=(0.65, 1.0), ylim=(0.65, 1.01),
               xlabel="Nominal interval coverage", ylabel="Empirical validation coverage",
               title=f"{target.upper()} — diagnostic coverage sensitivity")
        ax.legend(fontsize=8)
        fig.tight_layout()
        mlflow.log_figure(fig, "plots/validation_coverage_by_confidence.png")
    finally:
        plt.close(fig)


def log_model_artifacts(
    model: SplitConformalRegressor,
    validation: PotencyIntervalScores,
    sensitivity: pd.DataFrame,
) -> None:
    with TemporaryDirectory() as tmp:
        directory = Path(tmp)
        forest_path = directory / f"{model.target}_stage_d3_forest.joblib"
        joblib.dump(model.forest, forest_path)
        mlflow.log_artifact(str(forest_path), artifact_path="models")
        mlflow.log_dict({
            "target": model.target,
            "alpha": model.alpha,
            "half_width_pic50": model.half_width,
            "calibration_order_statistic": model.calibration_order_statistic,
            "n_fit_labels": model.n_fit_labels,
            "n_calibration_labels": model.n_calibration_labels,
            "assignment_sha256": model.assignment_sha256,
            "dataset_sha256": model.dataset_sha256,
            "manifest_sha256": model.manifest_sha256,
            "train_feature_sha256": model.feature_sha256,
            "seed_used": model.seed_used,
            "config": asdict(model.config),
        }, "models/conformal_wrapper_metadata.json")
        for role, scores in (
            ("calibration", model.calibration_scores),
            ("validation", validation),
        ):
            csv_path = directory / f"{model.target}_{role}_interval_predictions.csv"
            _interval_frame(scores).to_csv(csv_path, index=False)
            mlflow.log_artifact(str(csv_path), artifact_path=f"predictions/{role}")
        curve_path = directory / f"{model.target}_coverage_sensitivity.csv"
        sensitivity.to_csv(curve_path, index=False)
        mlflow.log_artifact(str(curve_path), artifact_path="diagnostics")


def main() -> None:
    args = parse_args()
    pipeline = load_modeling_pipeline(
        args.dataset, args.manifest, args.summary, n_splits=args.folds
    )
    split = make_scaffold_calibration_split(
        pipeline,
        calibration_fraction=0.20,
        threshold_fraction=0.20,
        seed=719,
        minimum_target_labels=10,
        minimum_dual_per_class=50,
        max_attempts=1,
    )
    split_audit = verify_frozen_assignment(split, pipeline, args)
    _verify_target_counts(split, pipeline)
    # This fit-only view is the SAME selection boundary used by the corrected
    # Optuna run. Do not pass the full 8,692-molecule pipeline to read_selections.
    fit_only_pipeline = build_fit_only_tuning_pipeline(
        pipeline, split, n_splits=args.folds,
    )
    if len(fit_only_pipeline.data.train) != 5364:
        raise ValueError("Unexpected fit-only Optuna molecule count")
    fit_only_bank = build_morgan_feature_bank(fit_only_pipeline.data.train)
    selections, selection_provenance = read_selections(
        fit_only_pipeline, fit_only_bank, args
    )
    _verify_fit_only_optuna_selections(selections, selection_provenance)

    # Conformal fitting/calibration needs both roles in the original train bank.
    # The conformal module indexes only split.fit and split.calibration labels.
    train_bank = build_morgan_feature_bank(pipeline.data.train)
    validation_bank = build_morgan_feature_bank(pipeline.data.validation)

    print("STAGE D.3 — FIT-ONLY OPTUNA EGFR/HER2 CONFORMAL REGRESSION")
    print(f"Frozen assignment SHA-256: {split.assignment_sha256}")
    print(f"Original M2.1 training partition: {len(pipeline.data.train)} molecules")
    print(f"Optuna / RF fit subset: {len(split.fit)} molecules")
    print(f"Reserved conformal calibration subset: {len(split.calibration)} molecules")
    print(f"Reserved threshold subset: {len(split.threshold)} molecules")
    print(f"Development-validation molecules: {len(pipeline.data.validation)}")
    for target in TARGETS:
        counts = EXPECTED_LABEL_COUNTS[target]
        print(f"{target.upper()}: fit={counts['fit']}, calibration={counts['calibration']}, "
              f"threshold RESERVED={counts['threshold']}, validation={counts['validation']}")
        print(f"  Optuna source trial {selection_provenance[target]['best_trial']}, "
              f"CV MAE {selection_provenance[target]['cv_mean_score']:.4f}")

    configure_mlflow(args)
    with mlflow.start_run(run_name=args.run_name) as parent:
        commit, dirty = git_state()
        mlflow.set_tags({
            "milestone": "M2.2_stage_D3",
            "training_protocol": "seed719_scaffold_disjoint_split_conformal",
            "calibration_method": "split_conformal_absolute_residual",
            "model_variant": "fit_only_optuna_reference_first_200tree_RF",
            "fit_only_optuna_parent_run_id": FIT_ONLY_OPTUNA_PARENT_RUN_ID,
            "threshold_role_used": "false",
            "development_validation_used_for_fitting": "false",
            "test_labels_loaded": "false",
            "stage_b_parent_run_id": STAGE_B_RUN_ID,
            "stage_c_split_audit_run_id": STAGE_C_AUDIT_RUN_ID,
            "stage_c_seed_selection_run_id": STAGE_C_SELECTION_RUN_ID,
            "stage_d1_dual_rf_calibration_run_id": STAGE_D1_RUN_ID,
            "git_commit": commit,
            "git_dirty": dirty,
        })
        mlflow.log_params({
            "dataset_sha256": pipeline.data.dataset_sha256,
            "manifest_sha256": pipeline.data.manifest_sha256,
            "curation_summary_sha256": _file_sha256(args.summary),
            "stage_c_assignment_sha256": split.assignment_sha256,
            "stage_c_split_manifest_sha256": _file_sha256(args.split_manifest),
            "stage_c_split_audit_sha256": _file_sha256(args.split_audit),
            "train_feature_sha256": train_bank.fingerprint_sha256,
            "fit_only_optuna_feature_sha256": fit_only_bank.fingerprint_sha256,
            "validation_feature_sha256": validation_bank.fingerprint_sha256,
            "optuna_source_parent_run_id": FIT_ONLY_OPTUNA_PARENT_RUN_ID,
            "fit_only_tuning_molecules": len(fit_only_pipeline.data.train),
            "alpha": args.alpha,
            "target_coverage": 1.0 - args.alpha,
            "rf_trees": args.trees,
            "rf_seed": selections["egfr"].seed,
            "rf_n_jobs": args.n_jobs,
            "split_seed": split.seed_used,
        })
        _log_role_identities(pipeline, split)
        for path in (args.manifest, args.summary, args.split_manifest, args.split_audit):
            mlflow.log_artifact(str(path), artifact_path="provenance")
        mlflow.log_dict({
            "frozen_assignment_sha256": FROZEN_ASSIGNMENT_SHA256,
            "stage_c_role_audits": split_audit["audits"],
            "optuna_selections": {t: selection_provenance[t] for t in TARGETS},
            "stage_b_parent_run_id": STAGE_B_RUN_ID,
            "stage_c_split_audit_run_id": STAGE_C_AUDIT_RUN_ID,
            "stage_c_seed_selection_run_id": STAGE_C_SELECTION_RUN_ID,
            "stage_d1_dual_rf_calibration_run_id": STAGE_D1_RUN_ID,
            "interval_method": "global_symmetrical_absolute_residual_split_conformal",
            "target_coverage_selected_before_validation": True,
            "calibration_quantile_uses_only_stage_c_calibration_role": True,
            "threshold_role_reserved": True,
            "final_test_labels_loaded": False,
            "stage_b_full_training_forests_reused": False,
            "optuna_selection_used_fit_role_only": True,
            "calibration_or_threshold_labels_used_for_optuna_selection": False,
            "label_informed_seed_719_selection_caveat": True,
            "previous_development_validation_exploration_caveat": True,
            "interpretation_caveat": (
                "Nominal marginal coverage relies on exchangeability. "
                "Scaffold-disjoint validation and label-informed seed selection "
                "may violate assumptions; report empirical coverage only. "
                "Coverage-sensitivity plot is diagnostic, NOT used to pick alpha."
            ),
        }, "provenance/stage_d3_protocol.json")

        rows = []
        parent_metrics = {}
        for target in TARGETS:
            config = selections[target]
            with mlflow.start_run(run_name=f"{target.upper()}_OptunaRF_Conformal", nested=True):
                mlflow.set_tags({
                    "target": target,
                    "calibration_method": "absolute_residual_split_conformal",
                    "validation_only_evaluation": "true",
                    "source_optuna_study": selection_provenance[target]["study_name"],
                    "source_optuna_trial": str(selection_provenance[target]["best_trial"]),
                })
                mlflow.log_params({
                    "assignment_sha256": split.assignment_sha256,
                    "target": target,
                    "alpha": args.alpha,
                    "target_coverage": 1.0 - args.alpha,
                    "fit_labels_expected": EXPECTED_LABEL_COUNTS[target]["fit"],
                    "calibration_labels_expected": EXPECTED_LABEL_COUNTS[target]["calibration"],
                    "threshold_labels_reserved": EXPECTED_LABEL_COUNTS[target]["threshold"],
                    "source_optuna_study": selection_provenance[target]["study_name"],
                    "source_optuna_best_trial": selection_provenance[target]["best_trial"],
                })
                mlflow.log_dict(asdict(config), "provenance/selected_rf_config.json")
                model = fit_split_conformal_regression(
                    pipeline, split, train_bank, config, target,
                    alpha=args.alpha,
                    expected_assignment_sha256=FROZEN_ASSIGNMENT_SHA256,
                )
                validation = evaluate_conformal_development_validation(
                    model, pipeline, validation_bank
                )
                if model.n_fit_labels != EXPECTED_LABEL_COUNTS[target]["fit"]:
                    raise ValueError(f"Unexpected {target} fitting-label count")
                if model.n_calibration_labels != EXPECTED_LABEL_COUNTS[target]["calibration"]:
                    raise ValueError(f"Unexpected {target} calibration-label count")
                if validation.metrics.n != EXPECTED_LABEL_COUNTS[target]["validation"]:
                    raise ValueError(f"Unexpected {target} validation-label count")
                for role, scores in (("calibration", model.calibration_scores),
                                     ("validation", validation)):
                    rows.append(_metric_row(target, role, scores))
                    mlflow.log_metrics({
                        f"{role}_mae": scores.metrics.mae,
                        f"{role}_coverage": scores.metrics.empirical_coverage,
                        f"{role}_mean_interval_width_pic50": scores.metrics.mean_interval_width,
                        f"{role}_median_interval_width_pic50": scores.metrics.median_interval_width,
                        f"{role}_n_labels": scores.metrics.n,
                    })
                mlflow.log_metrics({
                    "conformal_half_width_pic50": model.half_width,
                    "calibration_order_statistic": model.calibration_order_statistic,
                    "fit_n_labels": model.n_fit_labels,
                    "calibration_n_labels": model.n_calibration_labels,
                    "threshold_n_labels_reserved": EXPECTED_LABEL_COUNTS[target]["threshold"],
                })
                sensitivity = _confidence_sensitivity(model, validation)
                log_model_artifacts(model, validation, sensitivity)
                _plot_parity_with_intervals(target, validation)
                _plot_residuals(target, model, validation)
                _plot_coverage_sensitivity(target, sensitivity)
                parent_metrics[f"{target}_validation_coverage"] = validation.metrics.empirical_coverage
                parent_metrics[f"{target}_validation_mean_interval_width_pic50"] = (
                    validation.metrics.mean_interval_width
                )
                parent_metrics[f"{target}_validation_mae"] = validation.metrics.mae
                parent_metrics[f"{target}_interval_half_width_pic50"] = model.half_width
                print(f"{target.upper()}  validation MAE={validation.metrics.mae:.4f}; "
                      f"coverage={validation.metrics.empirical_coverage:.3f}; "
                      f"mean width={validation.metrics.mean_interval_width:.3f}; "
                      f"calibrated half-width={model.half_width:.3f}")
        mlflow.log_metrics(parent_metrics)
        mlflow.log_dict({"rows": rows}, "comparison/conformal_metrics.json")
        with TemporaryDirectory() as tmp:
            comparison_csv = Path(tmp) / "stage_d3_conformal_metrics.csv"
            pd.DataFrame(rows).to_csv(comparison_csv, index=False)
            mlflow.log_artifact(str(comparison_csv), artifact_path="comparison")
        print(f"\nStage D.3 parent MLflow run: {parent.info.run_id}")
        print("Threshold subset untouched; final test labels were not loaded.")
        print("Diagnostic coverage curve does not select alpha or alter the fitted models.")


if __name__ == "__main__":
    main()
