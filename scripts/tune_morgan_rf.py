"""Tune Morgan RFs with scaffold CV on Stage C fitting molecules ONLY.

This corrected run reserves seed-719 calibration and threshold molecules
before Optuna selection. Development validation and locked test stay excluded.
Historical 8,692-molecule studies are not resumed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import mlflow
import optuna
import pandas as pd
import rdkit
import sklearn
from dual_target.calibration.scaffold_calibration import make_scaffold_calibration_split
from dual_target.data.modeling_dataset import sha256_file
from dual_target.data.pipeline import (
    FROZEN_STAGE_C_ASSIGNMENT_SHA256,
    build_fit_only_tuning_pipeline,
    load_modeling_pipeline,
    select_task_records,
)
from dual_target.evaluation.scaffold_cv import fold_manifest
from dual_target.features.morgan_pipeline import build_morgan_feature_bank
from dual_target.models.random_forest import RFConfig
from dual_target.training.optuna_rf import optimize_rf
from optuna.trial import TrialState

DEFAULT_DATASET = Path("data/interim/M2_1_scaffold_split_molecules.csv")
DEFAULT_MANIFEST = Path("data/interim/M2_1_scaffold_split_manifest.csv")
DEFAULT_SUMMARY = Path("reports/audits/M2_1_scaffold_split_summary.json")
DEFAULT_MLFLOW_DB = Path("reports/mlflow/mlflow.db")
DEFAULT_OPTUNA_DB = Path("reports/optuna/rf_fit_only_seed719.db")
DEFAULT_STAGE_C_MANIFEST = Path(
    "reports/calibration/M2_2_C3_scaffold_split_2b99214bce37.csv"
)
DEFAULT_STAGE_C_AUDIT = Path(
    "reports/calibration/M2_2_C3_scaffold_split_2b99214bce37.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--stage-c-manifest", type=Path, default=DEFAULT_STAGE_C_MANIFEST)
    parser.add_argument("--stage-c-audit", type=Path, default=DEFAULT_STAGE_C_AUDIT)
    parser.add_argument("--tasks", nargs="+", choices=("egfr", "her2", "dual"),
                        default=("egfr", "her2", "dual"))
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--run-name", default="M2_2_RF_Optuna_FitOnly_Seed719")
    parser.add_argument("--experiment", default="EGFR_HER2_M2_Scaffold_Benchmark")
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--study-db", type=Path, default=DEFAULT_OPTUNA_DB)
    parser.add_argument("--study-prefix", default="M2_2_RF_fit_only_seed719")
    parser.add_argument("--reference-first", action="store_true")
    args = parser.parse_args()
    if args.trials < 1 or args.trees < 1:
        parser.error("--trials and --trees must be positive")
    if args.folds < 2:
        parser.error("--folds must be at least 2")
    if args.n_jobs == 0:
        parser.error("--n-jobs must not be zero")
    if len(set(args.tasks)) != len(args.tasks):
        parser.error("--tasks must not contain duplicates")
    return args


def sqlite_uri(db_path: Path) -> str:
    resolved = db_path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{resolved.as_posix()}"



def verify_frozen_stage_c_artifacts(split, source_pipeline, args) -> None:
    """Fail closed if frozen split, manifest CSV, or audit JSON changed."""
    if split.assignment_sha256 != FROZEN_STAGE_C_ASSIGNMENT_SHA256:
        raise ValueError("Wrong Stage C assignment SHA-256")
    if split.seed_used != 719:
        raise ValueError("Expected frozen Stage C seed 719")
    if (len(split.fit), len(split.calibration), len(split.threshold)) != (
        5364, 1658, 1670,
    ):
        raise ValueError("Frozen Stage C role counts changed")
    if (
        split.fit_audit.paired_labels,
        split.calibration_audit.paired_labels,
        split.threshold_audit.paired_labels,
    ) != (697, 223, 164):
        raise ValueError("Frozen Stage C paired-label counts changed")

    # Verify byte-for-byte equality against the independently audited CSV.
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=("structure_id", "scaffold_group_id", "role"),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(split.manifest_rows())
    saved_csv = args.stage_c_manifest.read_bytes()
    if saved_csv != stream.getvalue().encode("utf-8"):
        raise ValueError("Stage C role CSV differs from the frozen assignments")

    audit = json.loads(args.stage_c_audit.read_text(encoding="utf-8"))
    details = audit.get("split", {})
    source = audit.get("source", {})
    if (
        details.get("seed_used") != 719
        or details.get("assignment_sha256") != split.assignment_sha256
        or details.get("manifest_csv_sha256") != hashlib.sha256(saved_csv).hexdigest()
        or source.get("dataset_sha256") != source_pipeline.data.dataset_sha256
        or source.get("manifest_sha256") != source_pipeline.data.manifest_sha256
        or source.get("curation_summary_sha256") != sha256_file(args.summary)
    ):
        raise ValueError("Stage C audit provenance differs from frozen source")
    role_audits = {row["role"]: row for row in audit.get("audits", [])}
    for role in ("fit", "calibration", "threshold"):
        recorded = role_audits.get(role)
        expected = asdict(getattr(split, f"{role}_audit"))
        if recorded is None or any(recorded.get(k) != v for k, v in expected.items()):
            raise ValueError(f"Stage C audit counts changed for role {role}")


def configure_mlflow(args: argparse.Namespace) -> None:
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    elif not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(sqlite_uri(DEFAULT_MLFLOW_DB))
    mlflow.set_experiment(args.experiment)


def git_state() -> tuple[str, str]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"],
            check=True, capture_output=True, text=True,
        ).stdout.strip())
        return commit, str(dirty)
    except (OSError, subprocess.CalledProcessError):
        return "unknown", "unknown"


def log_input_ids(records, source: Path, name: str) -> None:
    """Register molecule identities without uploading potency labels."""
    frame = pd.DataFrame({
        "molecule_id": [item.structure_id for item in records],
        "scaffold_id": [item.scaffold_group_id for item in records],
        "egfr_mask": [str(item.egfr_mask) for item in records],
        "her2_mask": [str(item.her2_mask) for item in records],
    })
    dataset = mlflow.data.from_pandas(
        frame, name=name, source=source.resolve().as_uri(),
    )
    mlflow.log_input(dataset, context="training")


def log_parent_provenance(pipeline, bank, args: argparse.Namespace) -> None:
    commit, dirty = git_state()
    mlflow.set_tags({
        "model_family": "rf_optuna",
        "evaluation": "fit_only_scaffold_cv",
        "git_commit": commit,
        "calibration_labels_used_for_selection": "false",
        "threshold_labels_used_for_selection": "false",
        "stage_c_assignment_sha256": pipeline.stage_c_assignment_sha256,
        "git_dirty": dirty,
        "test_labels_used": "false",
        "development_validation_used_for_selection": "false",
        "reference_first": str(args.reference_first).lower(),
    })
    mlflow.log_params({
        "dataset_sha256": pipeline.data.dataset_sha256,
        "manifest_sha256": pipeline.data.manifest_sha256,
        "feature_sha256": bank.fingerprint_sha256,
        "summary_sha256": sha256_file(args.summary),
        "python_version": platform.python_version(),
        "rdkit_version": rdkit.__version__,
        "sklearn_version": sklearn.__version__,
        "optuna_version": optuna.__version__,
        "n_folds": args.folds,
        "n_trials_added_per_study": args.trials,
        "reference_first": args.reference_first,
        "n_estimators": args.trees,
        "seed": 714,
        "stage_c_seed": pipeline.stage_c_seed,
        "fit_only_molecules": len(pipeline.data.train),
    })
    log_input_ids(pipeline.data.train, args.dataset, "M2_2_stage_c_fit_only_training")
    mlflow.log_artifact(str(args.manifest), artifact_path="provenance")
    mlflow.log_artifact(str(args.stage_c_manifest), artifact_path="provenance")
    mlflow.log_artifact(str(args.stage_c_audit), artifact_path="provenance")
    mlflow.log_artifact(str(args.summary), artifact_path="provenance")
    mlflow.log_dict({
        "dataset_sha256": pipeline.data.dataset_sha256,
        "manifest_sha256": pipeline.data.manifest_sha256,
        "feature_sha256": bank.fingerprint_sha256,
        "summary_sha256": sha256_file(args.summary),
        "data_source": str(args.dataset.resolve()),
        "model_selection_split": "Stage C fit only (5,364 molecules)",
        "stage_c_assignment_sha256": pipeline.stage_c_assignment_sha256,
        "stage_c_seed": pipeline.stage_c_seed,
        "calibration_labels_used_for_selection": False,
        "threshold_labels_used_for_selection": False,
        "test_labels_used": False,
    }, "provenance/checksums.json")
    with TemporaryDirectory() as temporary:
        csv_path = Path(temporary) / "internal_scaffold_folds.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=("structure_id", "scaffold_group_id", "holdout_fold"),
            )
            writer.writeheader()
            writer.writerows(asdict(row) for row in fold_manifest(pipeline.folds))
        mlflow.log_artifact(str(csv_path), artifact_path="provenance")


def make_trial_callback(task: str):
    """Log a completed Optuna trial as a nested MLflow run."""
    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        with mlflow.start_run(
            run_name=f"{task}_trial_{trial.number:04d}", nested=True,
        ):
            mlflow.set_tags({
                "model_family": "rf_optuna_trial",
                "task": task,
                "optuna_state": trial.state.name,
                "trial_role": trial.user_attrs.get("trial_role", "search"),
            })
            mlflow.log_params({
                "trial_number": trial.number,
                **{key: str(value) for key, value in trial.params.items()},
            })
            if trial.value is not None:
                mlflow.log_metric("cv_objective", float(trial.value))
            for key, value in trial.user_attrs.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    mlflow.log_metric(key, float(value))
    return callback


def log_study_artifacts(result) -> None:
    """Persist study history and plot objective changes across trials."""
    study = result.study
    completed = [
        trial for trial in study.trials
        if trial.state == TrialState.COMPLETE and trial.value is not None
    ]
    if not completed:
        raise RuntimeError("Optuna study has no completed trials")

    mlflow.log_dict(asdict(result.best_config), "study/best_config.json")
    mlflow.log_dict({
        "task": result.task,
        "metric": result.metric_name,
        "best_score": result.best_score,
        "best_trial_number": study.best_trial.number,
        "best_parameters": study.best_trial.params,
        "study_name": study.study_name,
        "study_provenance": study.user_attrs,
        "total_completed_trials": len(completed),
        "reference_first": study.user_attrs.get("reference_first", False),
    }, "study/summary.json")

    with TemporaryDirectory() as temporary:
        csv_path = Path(temporary) / "optuna_trials.csv"
        fieldnames = ["number", "state", "objective", "parameters", "metrics"]
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows({
                "number": trial.number,
                "state": trial.state.name,
                "objective": trial.value,
                "parameters": str(trial.params),
                "metrics": str(trial.user_attrs),
            } for trial in study.trials)
        mlflow.log_artifact(str(csv_path), artifact_path="study")

    values = [float(trial.value) for trial in completed]
    numbers = [trial.number for trial in completed]
    best_so_far = []
    running_best = values[0]
    for value in values:
        if result.task == "dual":
            running_best = max(running_best, value)
        else:
            running_best = min(running_best, value)
        best_so_far.append(running_best)

    fig, ax = plt.subplots(figsize=(7, 4))
    try:
        ax.plot(numbers, values, "o-", label="Trial CV score")
        ax.step(numbers, best_so_far, where="post", label="Best so far")
        ax.set(
            xlabel="Optuna trial number",
            ylabel=f"Mean scaffold-CV {result.metric_name}",
            title=f"{result.task.upper()} Random Forest optimization",
        )
        ax.legend()
        fig.tight_layout()
        mlflow.log_figure(fig, "plots/optuna_history.png")
    finally:
        plt.close(fig)


def main() -> None:
    args = parse_args()
    original = load_modeling_pipeline(
        args.dataset, args.manifest, args.summary, n_splits=args.folds,
    )
    split = make_scaffold_calibration_split(
        original,
        seed=719,
        calibration_fraction=0.20,
        threshold_fraction=0.20,
        minimum_dual_per_class=50,
        max_attempts=1,
    )
    verify_frozen_stage_c_artifacts(split, original, args)
    pipeline = build_fit_only_tuning_pipeline(
        original, split, n_splits=args.folds,
    )
    if len(pipeline.data.train) != 5364:
        raise ValueError("Expected exactly 5,364 Stage C fitting molecules")
    bank = build_morgan_feature_bank(pipeline.data.train)
    config = RFConfig(n_estimators=args.trees, n_jobs=args.n_jobs, seed=714)

    for task in args.tasks:
        for audit in pipeline.fold_audits:
            ready = {
                "egfr": audit.egfr_regression_ready,
                "her2": audit.her2_regression_ready,
                "dual": audit.dual_classification_ready,
            }[task]
            if not ready:
                raise ValueError(f"Fold {audit.number} not ready for task {task}")

    configure_mlflow(args)
    storage_uri = sqlite_uri(args.study_db)
    print(f"Fit-only tuning molecules: {len(pipeline.data.train)}")
    print(f"Reserved calibration molecules: {len(split.calibration)}")
    print(f"Reserved threshold molecules: {len(split.threshold)}")
    print(f"Stage C assignment SHA-256: {pipeline.stage_c_assignment_sha256}")
    print(f"Fingerprint SHA-256: {bank.fingerprint_sha256}")
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Optuna study storage: {storage_uri}")

    with mlflow.start_run(run_name=args.run_name) as parent_run:
        log_parent_provenance(pipeline, bank, args)
        for task in args.tasks:
            eligible = select_task_records(pipeline.data.train, task)
            study_name = (
                f"{args.study_prefix}_{task}_{args.trees}trees_{args.folds}folds"
            )
            print(f"\nTuning {task.upper()} ({len(eligible)} training labels)")
            with mlflow.start_run(run_name=f"Optuna_RF_{task}", nested=True):
                mlflow.set_tags({
                    "task": task,
                    "model_family": "rf_optuna_study",
                    "evaluation": "fit_only_scaffold_cv",
                    "stage_c_assignment_sha256": pipeline.stage_c_assignment_sha256,
                    "reference_first": str(args.reference_first).lower(),
                })
                log_input_ids(eligible, args.dataset, f"M2_1_{task}_training")
                mlflow.log_params({
                    "dataset_sha256": pipeline.data.dataset_sha256,
                    "manifest_sha256": pipeline.data.manifest_sha256,
                    "feature_sha256": bank.fingerprint_sha256,
                    "study_name": study_name,
                    "stage_c_assignment_sha256": pipeline.stage_c_assignment_sha256,
                    "n_training_labels": len(eligible),
                    "reference_first": args.reference_first,
                })
                result = optimize_rf(
                    pipeline=pipeline,
                    bank=bank,
                    task=task,
                    base_config=config,
                    n_trials=args.trials,
                    study_name=study_name,
                    storage=storage_uri,
                    callbacks=(make_trial_callback(task),),
                    reference_first=args.reference_first,
                )
                result.study.set_user_attr(
                    "stage_c_assignment_sha256",
                    pipeline.stage_c_assignment_sha256,
                )
                mlflow.log_metric(f"best_cv_{result.metric_name}", result.best_score)
                mlflow.log_metric("pooled_oof_score_best_trial", float(
                    result.study.best_trial.user_attrs["pooled_oof_score"]
                ))
                mlflow.log_params({
                    f"best_{key}": str(value)
                    for key, value in result.study.best_trial.params.items()
                })
                log_study_artifacts(result)
                print(f"Best mean CV {result.metric_name}: {result.best_score:.4f}")
                print(f"Best settings: {result.study.best_trial.params}")
        print(f"\nParent MLflow run ID: {parent_run.info.run_id}")


if __name__ == "__main__":
    main()
