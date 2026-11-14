"""Stage B: compare Optuna-selected Morgan RFs with matched untuned RFs.

Read only the frozen M2.1 train and development-validation partitions. Optuna
study selection is validated against training dataset, feature, and fold hashes
*before* any model is fitted. Validation labels are used for reporting only:
no tuning, calibration fitting, or decision-threshold selection occurs here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
from dual_target.data.modeling_dataset import sha256_file
from dual_target.data.pipeline import load_modeling_pipeline
from dual_target.evaluation.scaffold_cv import fold_manifest
from dual_target.features.morgan_pipeline import build_morgan_feature_bank
from dual_target.models.random_forest import RFConfig
from dual_target.training.rf_pipeline import (
    RFStageBResult,
    fit_and_evaluate_selected_rf,
)
from optuna.study import StudyDirection
from optuna.trial import TrialState
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import brier_score_loss, log_loss, precision_recall_curve

TASKS = ("egfr", "her2", "dual")
DEFAULT_DATASET = Path("data/interim/M2_1_scaffold_split_molecules.csv")
DEFAULT_MANIFEST = Path("data/interim/M2_1_scaffold_split_manifest.csv")
DEFAULT_SUMMARY = Path("reports/audits/M2_1_scaffold_split_summary.json")
DEFAULT_MLFLOW_DB = Path("reports/mlflow/mlflow.db")
DEFAULT_OPTUNA_DB = Path("reports/optuna/rf_studies.db")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--study-db", type=Path, default=DEFAULT_OPTUNA_DB)
    parser.add_argument("--study-prefix", default="M2_2_RF_scaffold")
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-completed-trials", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=None,
                        help="Override forest worker count; defaults to recorded study value")
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--experiment", default="EGFR_HER2_M2_Scaffold_Benchmark")
    parser.add_argument("--run-name", default="M2_2_StageB_Optimized_vs_Reference_RF")
    args = parser.parse_args()
    if args.trees < 1 or args.folds < 2 or args.min_completed_trials < 1:
        parser.error("--trees and --min-completed-trials must be positive; --folds >= 2")
    if args.n_jobs == 0:
        parser.error("--n-jobs cannot be zero")
    return args


def sqlite_uri(path: Path, *, create_parent: bool) -> str:
    absolute = path.resolve()
    if create_parent:
        absolute.parent.mkdir(parents=True, exist_ok=True)
    elif not absolute.is_file():
        raise FileNotFoundError(f"Optuna study database does not exist: {absolute}")
    return f"sqlite:///{absolute.as_posix()}"


def fold_assignments_sha256(pipeline) -> str:
    """Exactly the manifest digest used by training/optuna_rf.py."""
    rows = [asdict(row) for row in fold_manifest(pipeline.folds)]
    serialized = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def read_selections(pipeline, bank, args: argparse.Namespace):
    """Verify studies and reconstruct configurations without reading validation scores."""
    storage = sqlite_uri(args.study_db, create_parent=False)
    expected = {
        "dataset_sha256": pipeline.data.dataset_sha256,
        "manifest_sha256": pipeline.data.manifest_sha256,
        "feature_sha256": bank.fingerprint_sha256,
        "fold_assignments_sha256": fold_assignments_sha256(pipeline),
        "evaluation": "training_only_scaffold_group_cv",
    }
    configs = {}
    metadata = {}
    for task in TASKS:
        name = f"{args.study_prefix}_{task}_{args.trees}trees_{args.folds}folds"
        study = optuna.load_study(study_name=name, storage=storage)
        metric = "average_precision" if task == "dual" else "mae"
        direction = StudyDirection.MAXIMIZE if task == "dual" else StudyDirection.MINIMIZE
        if study.direction != direction:
            raise ValueError(f"{name}: unexpected optimization direction")
        for field, value in {**expected, "task": task, "metric_name": metric}.items():
            if study.user_attrs.get(field) != value:
                raise ValueError(f"{name}: incompatible/missing study provenance: {field}")

        base_values = study.user_attrs.get("base_config")
        if not isinstance(base_values, dict):
            raise TypeError(f"{name}: missing or invalid RFConfig provenance")
        base = RFConfig(**base_values)
        if base.n_estimators != args.trees:
            raise ValueError(f"{name}: tree count disagrees with recorded settings")
        if (base.radius, base.n_bits, base.include_chirality) != (
            bank.radius, bank.n_bits, bank.include_chirality
        ):
            raise ValueError(f"{name}: fingerprint settings disagree with training bank")

        finished = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
        if len(finished) < args.min_completed_trials:
            raise ValueError(
                f"{name}: only {len(finished)} completed trials; "
                f"expected >= {args.min_completed_trials}"
            )
        best = study.best_trial
        required = {"max_depth", "min_samples_leaf", "min_samples_split", "max_features"}
        if set(best.params) != required:
            raise ValueError(f"{name}: unexpected Optuna search space: {best.params}")
        selected = replace(
            base,
            max_depth=best.params["max_depth"],
            min_samples_leaf=int(best.params["min_samples_leaf"]),
            min_samples_split=int(best.params["min_samples_split"]),
            max_features=best.params["max_features"],
            n_jobs=args.n_jobs if args.n_jobs is not None else base.n_jobs,
        )
        configs[task] = selected
        metadata[task] = {
            "study_name": name,
            "completed_trials": len(finished),
            "best_trial": best.number,
            "cv_metric": metric,
            "cv_mean_score": float(best.value),
            "cv_pooled_oof_score": best.user_attrs.get("pooled_oof_score"),
            "cv_mean_train_score": best.user_attrs.get("mean_train_score"),
            "best_trial_parameters": dict(best.params),
            "recorded_base_config": base_values,
            "evaluated_config": asdict(selected),
        }
    return configs, metadata


def tracking_setup(args: argparse.Namespace) -> None:
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    elif not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(sqlite_uri(DEFAULT_MLFLOW_DB, create_parent=True))
    mlflow.set_experiment(args.experiment)


def git_state() -> tuple[str, str]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        changed = subprocess.run(
            ["git", "status", "--porcelain"], check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        return commit, str(bool(changed))
    except (OSError, subprocess.CalledProcessError):
        return "unknown", "unknown"


def log_dataset_identity(data, source: Path) -> None:
    """Record input lineage without uploading potency labels or the full CSV."""
    for split_name in ("train", "validation"):
        records = getattr(data, split_name)
        frame = pd.DataFrame({
            "molecule_id": [record.structure_id for record in records],
            "scaffold_id": [record.scaffold_group_id for record in records],
        })
        dataset = mlflow.data.from_pandas(
            frame, name=f"M2_1_{split_name}_identities",
            source=source.resolve().as_uri(),
        )
        mlflow.log_input(dataset, context=split_name)


def log_model_artifacts(fit) -> None:
    with TemporaryDirectory() as folder:
        path = Path(folder)
        model_path = path / f"{fit.task}_model.joblib"
        joblib.dump(fit.model, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="models")
        for split_name in ("train", "validation"):
            examples = getattr(fit, f"{split_name}_examples")
            preds = getattr(fit, f"{split_name}_predictions")
            label = examples.dual_active if fit.task == "dual" else examples.pic50
            output = path / f"{fit.task}_{split_name}_predictions.csv"
            pd.DataFrame({
                "molecule_id": examples.molecule_ids,
                "scaffold_id": examples.scaffold_ids,
                "observed": label,
                "predicted": preds,
            }).to_csv(output, index=False)
            mlflow.log_artifact(str(output), artifact_path=f"predictions/{split_name}")


def task_metric_rows(result: RFStageBResult, variant: str) -> list[dict]:
    rows = []
    for task in TASKS:
        fit = getattr(result, task)
        key = "average_precision" if task == "dual" else "mae"
        training = float(fit.train_metrics[key])
        validation = float(fit.validation_metrics[key])
        gap = training - validation if task == "dual" else validation - training
        row = {
            "variant": variant,
            "task": task,
            "primary_metric": key,
            "train_primary": training,
            "validation_primary": validation,
            "generalization_gap": gap,
            "n_train_labels": len(fit.train_predictions),
            "n_validation_labels": len(fit.validation_predictions),
            "validation_spearman": (
                fit.validation_metrics.get("spearman") if task != "dual" else None
            ),
        }
        rows.append(row)
    return rows


def log_variant(result: RFStageBResult, variant: str, configs) -> list[dict]:
    with mlflow.start_run(run_name=f"{variant}_200tree_RF", nested=True):
        mlflow.set_tags({
            "model_family": "morgan_random_forest",
            "variant": variant,
            "evaluation": "development_validation",
            "test_labels_used": "false",
            "uncalibrated_scores": "true",
        })
        for task in TASKS:
            fit = getattr(result, task)
            mlflow.log_dict(asdict(configs[task]), f"configs/{task}.json")
            for split, metrics in (("train", fit.train_metrics),
                                   ("validation", fit.validation_metrics)):
                mlflow.log_metrics({
                    f"{task}_{split}_{key}": float(value)
                    for key, value in metrics.items()
                    if value is not None and np.isfinite(float(value))
                })
            log_model_artifacts(fit)
        paired = result.paired_validation
        mlflow.log_metrics({
            "validation_paired_n": len(paired.molecule_ids),
            "validation_paired_prevalence": float(np.mean(paired.observed_dual_active)),
            "validation_paired_regression_ap": paired.regression_average_precision,
            "validation_paired_direct_classifier_ap": paired.classifier_average_precision,
            "validation_direct_classifier_brier": brier_score_loss(
                result.dual.validation_examples.dual_active,
                result.dual.validation_predictions,
            ),
            "validation_direct_classifier_log_loss": log_loss(
                result.dual.validation_examples.dual_active,
                result.dual.validation_predictions,
                labels=[0, 1],
            ),
        })
        with TemporaryDirectory() as folder:
            path = Path(folder) / "paired_validation_predictions.csv"
            pd.DataFrame({
                "molecule_id": paired.molecule_ids,
                "scaffold_id": paired.scaffold_ids,
                "observed_dual_active": paired.observed_dual_active,
                "predicted_egfr_pic50": paired.predicted_egfr_pic50,
                "predicted_her2_pic50": paired.predicted_her2_pic50,
                "regression_ranking_score": paired.regression_ranking_score,
                "classifier_score_uncalibrated": paired.classifier_score,
            }).to_csv(path, index=False)
            mlflow.log_artifact(str(path), artifact_path="predictions/validation")
    return task_metric_rows(result, variant)


def log_figure(fig, path: str) -> None:
    try:
        fig.tight_layout()
        mlflow.log_figure(fig, path)
    finally:
        plt.close(fig)


def plot_comparisons(reference: RFStageBResult, selected: RFStageBResult) -> None:
    for target in ("egfr", "her2"):
        ref_fit = getattr(reference, target)
        sel_fit = getattr(selected, target)
        obs = sel_fit.validation_examples.pic50
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(obs, ref_fit.validation_predictions, alpha=0.35, s=18,
                   label="Reference 200-tree RF")
        ax.scatter(obs, sel_fit.validation_predictions, alpha=0.35, s=18,
                   label="Optuna-selected 200-tree RF")
        values = np.concatenate([obs, ref_fit.validation_predictions,
                                 sel_fit.validation_predictions])
        ax.plot([values.min(), values.max()], [values.min(), values.max()], "k--",
                linewidth=1, label="Ideal")
        ax.set(xlabel="Measured pIC50", ylabel="Predicted pIC50",
               title=f"{target.upper()} — scaffold-held-out validation")
        ax.legend()
        log_figure(fig, f"plots/{target}_parity_comparison.png")

        fig, ax = plt.subplots(figsize=(6, 4))
        for fit, label in ((ref_fit, "Reference"), (sel_fit, "Optuna-selected")):
            resid = fit.validation_predictions - fit.validation_examples.pic50
            ax.scatter(fit.validation_predictions, resid, s=18, alpha=0.4,
                       label=label)
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set(xlabel="Predicted pIC50", ylabel="Residual (prediction - observed)",
               title=f"{target.upper()} — validation residuals")
        ax.legend()
        log_figure(fig, f"plots/{target}_residual_comparison.png")

        fig, ax = plt.subplots(figsize=(5, 4))
        values = [float(ref_fit.validation_metrics["mae"]),
                  float(sel_fit.validation_metrics["mae"])]
        bars = ax.bar(["Reference", "Selected"], values)
        ax.bar_label(bars, fmt="%.3f")
        ax.set(ylabel="Validation MAE (pIC50 units)",
               title=f"{target.upper()} — matched 200-tree comparison")
        log_figure(fig, f"plots/{target}_mae_comparison.png")

    for method in ("classifier", "regression"):
        fig, ax = plt.subplots(figsize=(6, 5))
        for result, label in ((reference, "Reference"), (selected, "Optuna-selected")):
            paired = result.paired_validation
            scores = (paired.classifier_score if method == "classifier"
                      else paired.regression_ranking_score)
            precision, recall, _ = precision_recall_curve(
                paired.observed_dual_active, scores
            )
            ap = (paired.classifier_average_precision if method == "classifier"
                  else paired.regression_average_precision)
            ax.step(recall, precision, where="post", label=f"{label} AP={ap:.3f}")
        prevalence = float(np.mean(selected.paired_validation.observed_dual_active))
        ax.axhline(prevalence, linestyle="--", color="gray",
                   label=f"Positive prevalence={prevalence:.3f}")
        ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.05),
               title=f"Dual activity — {method} ranking on paired validation")
        ax.legend()
        log_figure(fig, f"plots/dual_{method}_pr_comparison.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    for result, label in ((reference, "Reference"), (selected, "Optuna-selected")):
        fit = result.dual
        CalibrationDisplay.from_predictions(
            fit.validation_examples.dual_active,
            fit.validation_predictions,
            n_bins=8,
            strategy="quantile",
            name=label,
            ax=ax,
        )
    ax.set_title("Uncalibrated dual-RF reliability — diagnostic only")
    log_figure(fig, "plots/dual_uncalibrated_reliability.png")


def main() -> None:
    args = parse_args()
    pipeline = load_modeling_pipeline(
        args.dataset, args.manifest, args.summary, n_splits=args.folds
    )
    bank = build_morgan_feature_bank(pipeline.data.train)
    selected_configs, selections = read_selections(pipeline, bank, args)
    # Match tree count, seed, fingerprint representation, and worker count.
    # Only the four Optuna-searchable hyperparameters differ between variants.
    reference_configs = {
        task: RFConfig(
            radius=config.radius, n_bits=config.n_bits,
            include_chirality=config.include_chirality,
            n_estimators=config.n_estimators, seed=config.seed,
            n_jobs=config.n_jobs,
        )
        for task, config in selected_configs.items()
    }

    print(f"Frozen train molecules: {len(pipeline.data.train)}")
    print(f"Development-validation molecules: {len(pipeline.data.validation)}")
    print(f"Training feature SHA-256: {bank.fingerprint_sha256}")
    for task in TASKS:
        info = selections[task]
        print(f"{task.upper()} selection: trial {info['best_trial']} "
              f"(mean CV {info['cv_metric']}={info['cv_mean_score']:.4f})")

    tracking_setup(args)
    with mlflow.start_run(run_name=args.run_name) as parent:
        commit, dirty = git_state()
        mlflow.set_tags({
            "milestone": "M2.2_stage_B",
            "evaluation": "development_validation",
            "selection": "frozen_training_only_optuna_studies",
            "test_labels_used": "false",
            "validation_used_for_hyperparameter_selection": "false",
            "git_commit": commit,
            "git_dirty": dirty,
        })
        mlflow.log_params({
            "dataset_sha256": pipeline.data.dataset_sha256,
            "manifest_sha256": pipeline.data.manifest_sha256,
            "train_feature_sha256": bank.fingerprint_sha256,
            "fold_assignments_sha256": fold_assignments_sha256(pipeline),
            "summary_sha256": sha256_file(args.summary),
            "trees": args.trees,
            "cv_folds": args.folds,
            "min_completed_optuna_trials": args.min_completed_trials,
            "train_molecules": len(pipeline.data.train),
            "validation_molecules": len(pipeline.data.validation),
        })
        log_dataset_identity(pipeline.data, args.dataset)
        mlflow.log_artifact(str(args.manifest), artifact_path="provenance")
        mlflow.log_artifact(str(args.summary), artifact_path="provenance")
        mlflow.log_dict(selections, "provenance/optuna_selections.json")
        mlflow.log_dict({
            "dataset_path": str(args.dataset.resolve()),
            "dataset_sha256": pipeline.data.dataset_sha256,
            "manifest_sha256": pipeline.data.manifest_sha256,
            "train_fingerprint_sha256": bank.fingerprint_sha256,
            "study_db": str(args.study_db.resolve()),
            "test_partition_loaded": False,
            "validation_used_for_selection": False,
            "binary_activity_rule": "Both EGFR and HER2 pIC50 strictly > 6",
            "classifier_scores_calibrated": False,
            "decision_threshold_tuned": False,
        }, "provenance/stage_b_methodology.json")

        print("\nFitting matched untuned reference models...")
        reference = fit_and_evaluate_selected_rf(pipeline, bank, reference_configs)
        print("Fitting Optuna-selected models...")
        selected = fit_and_evaluate_selected_rf(pipeline, bank, selected_configs)
        rows = log_variant(reference, "Reference", reference_configs)
        rows.extend(log_variant(selected, "Optuna_selected", selected_configs))

        with TemporaryDirectory() as folder:
            output = Path(folder) / "stage_b_metrics.csv"
            with output.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            mlflow.log_artifact(str(output), artifact_path="comparison")
        for task in TASKS:
            for variant, result in (("reference", reference), ("selected", selected)):
                fit = getattr(result, task)
                primary = "average_precision" if task == "dual" else "mae"
                mlflow.log_metric(f"{variant}_{task}_validation_{primary}",
                                  float(fit.validation_metrics[primary]))
            baseline_fit = getattr(reference, task)
            tuned_fit = getattr(selected, task)
            print(f"{task.upper()}: reference validation "
                  f"{primary}={float(baseline_fit.validation_metrics[primary]):.4f}; "
                  f"selected={float(tuned_fit.validation_metrics[primary]):.4f}")
        for variant, result in (("reference", reference), ("selected", selected)):
            mlflow.log_metrics({
                f"{variant}_paired_regression_ap": result.paired_validation.regression_average_precision,
                f"{variant}_paired_classifier_ap": result.paired_validation.classifier_average_precision,
            })
        plot_comparisons(reference, selected)
        print(f"\nStage B parent MLflow run: {parent.info.run_id}")
        print("Development-validation scores are descriptive; no calibration or threshold was fitted.")


if __name__ == "__main__":
    main()
