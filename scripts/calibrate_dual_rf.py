"""Stage D.1: refit the fit-only Optuna-selected dual RF and sigmoid calibrator.

Verify the frozen seed-719 scaffold assignment. Train a fresh 200-tree RF
classifier on the 697 accepted paired labels in the 'fit' role using Trial 18
from the leakage-corrected Optuna study. Fit a sigmoid only on the disjoint
223 paired 'calibration' labels, then report diagnostics on M2.1 development
validation. The 164 paired 'threshold' labels and locked test stay untouched.

Optuna hyperparameters were selected on fit-only scaffold CV, not on the
calibration, threshold, or development-validation labels. Historical validation
exploration and label-informed selection of split seed 719 remain caveats.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from dual_target.calibration.probability import (
    DualProbabilityScores,
    evaluate_development_validation,
    fit_sigmoid_dual_calibration,
)
from dual_target.calibration.scaffold_calibration import (
    ScaffoldCalibrationSplit,
    make_scaffold_calibration_split,
)
from dual_target.data.modeling_dataset import sha256_file
from dual_target.data.pipeline import load_modeling_pipeline
from dual_target.features.morgan_pipeline import build_morgan_feature_bank
from dual_target.models.random_forest import RFConfig
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import precision_recall_curve

DEFAULT_DATASET = Path("data/interim/M2_1_scaffold_split_molecules.csv")
DEFAULT_MANIFEST = Path("data/interim/M2_1_scaffold_split_manifest.csv")
DEFAULT_SUMMARY = Path("reports/audits/M2_1_scaffold_split_summary.json")
DEFAULT_AUDIT = Path(
    "reports/calibration/M2_2_C3_scaffold_split_2b99214bce37.json"
)
DEFAULT_SPLIT_MANIFEST = Path(
    "reports/calibration/M2_2_C3_scaffold_split_2b99214bce37.csv"
)
DEFAULT_MLFLOW_DB = Path("reports/mlflow/mlflow.db")
FROZEN_ASSIGNMENT_SHA256 = (
    "2b99214bce37d30e1eed8f6e22984242492560b2999bb174a8d378604de9171c"
)
STAGE_B_RUN_ID = "12c9fc400b354f6da4e993e69e36b1cb"
STAGE_C_AUDIT_RUN_ID = "d2ca6b7898f34890a756c547c706bc56"
STAGE_C_SELECTION_RUN_ID = "5d013a8325904155ac3774e42e01a7f2"
OPTUNA_PARENT_RUN_ID = "2c748d4495ec47a4924d66043349709b"
OPTUNA_DUAL_STUDY = "M2_2_fit_only_reffirst_200trees_dual_200trees_5folds"
OPTUNA_BEST_TRIAL = 18
OPTUNA_BEST_CV_AP = 0.9076405916837615


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--split-audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--experiment", default="EGFR_HER2_M2_Scaffold_Benchmark")
    parser.add_argument(
        "--run-name", default="M2_2_StageD1_FitOnly_Optimized_Dual_RF_Sigmoid"
    )
    args = parser.parse_args()
    if args.trees != 200:
        parser.error("--trees must be 200 to match the selected fit-only Optuna trial")
    if args.n_jobs == 0:
        parser.error("--n-jobs must not be zero")
    return args


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Required frozen evidence not found: {path}")
    return sha256_file(path)


def _manifest_bytes(split: ScaffoldCalibrationSplit) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=("structure_id", "scaffold_group_id", "role"),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(split.manifest_rows())
    return stream.getvalue().encode("utf-8")


def verify_frozen_assignment(split: ScaffoldCalibrationSplit, pipeline, args) -> dict:
    """Fail closed unless split identity, saved CSV, audit, and counts all match."""
    if split.seed_used != 719 or split.assignment_sha256 != FROZEN_ASSIGNMENT_SHA256:
        raise ValueError("Reconstructed Stage C assignment differs from frozen seed 719")
    if len(split.fit) != 5364 or len(split.calibration) != 1658 or len(split.threshold) != 1670:
        raise ValueError("Unexpected Stage C molecule counts")
    if (
        split.fit_audit.paired_labels != 697
        or split.calibration_audit.paired_labels != 223
        or split.threshold_audit.paired_labels != 164
    ):
        raise ValueError("Unexpected Stage C paired-label counts")

    audit = json.loads(args.split_audit.read_text(encoding="utf-8"))
    source = audit.get("source", {})
    details = audit.get("split", {})
    if (
        source.get("dataset_sha256") != pipeline.data.dataset_sha256
        or source.get("manifest_sha256") != pipeline.data.manifest_sha256
        or source.get("curation_summary_sha256") != _file_sha256(args.summary)
        or details.get("assignment_sha256") != split.assignment_sha256
        or details.get("seed_used") != 719
        or details.get("requested_calibration_fraction_of_scaffolds") != 0.20
        or details.get("requested_threshold_fraction_of_scaffolds") != 0.20
        or details.get("max_attempts") != 1
    ):
        raise ValueError("Stage C audited split provenance differs from frozen files")
    saved_csv = args.split_manifest.read_bytes()
    expected_csv = _manifest_bytes(split)
    if saved_csv != expected_csv:
        raise ValueError("Saved Stage C manifest does not match reconstructed roles")
    if details.get("manifest_csv_sha256") != hashlib.sha256(saved_csv).hexdigest():
        raise ValueError("Stage C CSV checksum differs from audit")
    role_audits = {row["role"]: row for row in audit.get("audits", [])}
    for role in ("fit", "calibration", "threshold"):
        recorded = role_audits.get(role)
        if recorded is None:
            raise ValueError(f"Missing {role} role in frozen Stage C audit")
        computed = asdict(getattr(split, f"{role}_audit"))
        if any(recorded.get(key) != value for key, value in computed.items()):
            raise ValueError(f"Stage C audit count mismatch for {role}")
    return audit


def configure_mlflow(args: argparse.Namespace) -> None:
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    elif not os.environ.get("MLFLOW_TRACKING_URI"):
        path = DEFAULT_MLFLOW_DB.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(f"sqlite:///{path.as_posix()}")
    mlflow.set_experiment(args.experiment)


def git_state() -> tuple[str, str]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True,
            text=True, check=True,
        ).stdout.strip()
        changes = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True,
            text=True, check=True,
        ).stdout.strip()
        return commit, str(bool(changes))
    except (OSError, subprocess.CalledProcessError):
        return "unknown", "unknown"


def log_dataset_identities(pipeline, split, dataset_path: Path) -> None:
    """Log the actual fit/calibration/validation roles, without labels."""
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
            mlflow.data.from_pandas(
                frame, name=f"M2_2_stage_d1_{role}_identities",
                source=dataset_path.resolve().as_uri(),
            ),
            context="training" if role == "fit" else role,
        )


def _metric_dictionary(scores: DualProbabilityScores, prefix: str) -> dict[str, float]:
    raw = scores.raw_metrics
    calibrated = scores.calibrated_metrics
    metrics = {
        f"{prefix}_n": float(raw.n),
        f"{prefix}_positives": float(raw.positives),
        f"{prefix}_prevalence": raw.positive_prevalence,
    }
    for name in ("average_precision", "brier_score", "log_loss"):
        metrics[f"{prefix}_raw_{name}"] = float(getattr(raw, name))
        metrics[f"{prefix}_calibrated_{name}"] = float(getattr(calibrated, name))
    return metrics


def _prediction_frame(scores: DualProbabilityScores) -> pd.DataFrame:
    return pd.DataFrame({
        "molecule_id": scores.molecule_ids,
        "scaffold_id": scores.scaffold_ids,
        "observed_dual_active": scores.observed,
        "raw_probability": scores.raw_probabilities,
        "calibrated_probability": scores.calibrated_probabilities,
    })


def log_comparison_figures(validation: DualProbabilityScores) -> None:
    """Plot independent development-validation diagnostics, not test data."""
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    try:
        for probs, name in (
            (validation.raw_probabilities, "Uncalibrated RF"),
            (validation.calibrated_probabilities, "Sigmoid calibrated"),
        ):
            CalibrationDisplay.from_predictions(
                validation.observed, probs, n_bins=6, strategy="quantile",
                name=name, ax=ax,
            )
        ax.set_title("Dual-activity probability reliability — development validation")
        fig.tight_layout()
        mlflow.log_figure(fig, "plots/validation_reliability.png")
    finally:
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    try:
        for probs, metric, name in (
            (validation.raw_probabilities, validation.raw_metrics, "Uncalibrated"),
            (
                validation.calibrated_probabilities,
                validation.calibrated_metrics,
                "Sigmoid calibrated",
            ),
        ):
            precision, recall, _ = precision_recall_curve(validation.observed, probs)
            ax.step(
                recall, precision, where="post",
                label=f"{name}: AP={metric.average_precision:.3f}",
            )
        ax.axhline(
            np.mean(validation.observed), color="gray", linestyle="--",
            label="Validation prevalence",
        )
        ax.set(
            xlim=(0, 1), ylim=(0, 1.05),
            xlabel="Recall", ylabel="Precision",
            title="Dual-activity ranking — development validation",
        )
        ax.legend()
        fig.tight_layout()
        mlflow.log_figure(fig, "plots/validation_precision_recall.png")
    finally:
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    try:
        ax.hist(
            validation.raw_probabilities, bins=np.linspace(0, 1, 16),
            histtype="step", linewidth=1.8, label="Uncalibrated",
        )
        ax.hist(
            validation.calibrated_probabilities, bins=np.linspace(0, 1, 16),
            histtype="step", linewidth=1.8, label="Calibrated",
        )
        ax.set(
            xlim=(0, 1), xlabel="Predicted dual-active probability",
            ylabel="Number of paired molecules",
            title="Probability distributions — development validation",
        )
        ax.legend()
        fig.tight_layout()
        mlflow.log_figure(fig, "plots/validation_probability_histogram.png")
    finally:
        plt.close(fig)


def log_model_and_predictions(model, validation: DualProbabilityScores) -> None:
    with TemporaryDirectory() as directory:
        folder = Path(directory)
        rf_file = folder / "stage_d1_optimized_dual_rf.joblib"
        sigmoid_file = folder / "stage_d1_sigmoid.joblib"
        joblib.dump(model.forest, rf_file)
        joblib.dump(model.sigmoid, sigmoid_file)
        mlflow.log_artifact(str(rf_file), artifact_path="models")
        mlflow.log_artifact(str(sigmoid_file), artifact_path="models")

        for role, scores in (
            ("calibration", model.calibration_scores),
            ("validation", validation),
        ):
            output = folder / f"stage_d1_{role}_predictions.csv"
            _prediction_frame(scores).to_csv(output, index=False)
            mlflow.log_artifact(str(output), artifact_path=f"predictions/{role}")


def main() -> None:
    args = parse_args()
    pipeline = load_modeling_pipeline(
        args.dataset, args.manifest, args.summary, n_splits=5
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
    audit = verify_frozen_assignment(split, pipeline, args)
    # Selected solely by five-fold scaffold CV on the Stage C fit role.
    # Dual classifier: Optuna Trial 18 (200 trees, 20 total trials).
    config = RFConfig(
        n_estimators=args.trees,
        max_depth=20,
        min_samples_leaf=2,
        min_samples_split=4,
        max_features="sqrt",
        seed=714,
        n_jobs=args.n_jobs,
    )
    bank = build_morgan_feature_bank(pipeline.data.train)
    validation_bank = build_morgan_feature_bank(pipeline.data.validation)
    if (
        bank.radius, bank.n_bits, bank.include_chirality
    ) != (config.radius, config.n_bits, config.include_chirality):
        raise ValueError("Training fingerprint bank differs from optimized RF config")
    if (
        validation_bank.radius, validation_bank.n_bits, validation_bank.include_chirality
    ) != (config.radius, config.n_bits, config.include_chirality):
        raise ValueError("Validation fingerprint bank differs from optimized RF config")

    print("STAGE D.1 — FIT-ONLY OPTIMIZED DUAL RF AND SIGMOID CALIBRATION")
    print(f"Source Optuna study: {OPTUNA_DUAL_STUDY} (trial {OPTUNA_BEST_TRIAL})")
    print(f"Selected RF configuration: {asdict(config)}")
    print(f"Frozen assignment SHA-256: {split.assignment_sha256}")
    print(f"Train molecules (all roles): {len(pipeline.data.train)}")
    print(f"Fitting paired labels: {split.fit_audit.paired_labels}")
    print(f"Calibration paired labels: {split.calibration_audit.paired_labels}")
    print(f"Reserved threshold paired labels: {split.threshold_audit.paired_labels}")
    print(f"Development-validation molecules: {len(pipeline.data.validation)}")

    configure_mlflow(args)
    with mlflow.start_run(run_name=args.run_name) as parent:
        commit, dirty = git_state()
        mlflow.set_tags({
            "milestone": "M2.2_stage_D1",
            "training_protocol": "scaffold_disjoint_fit_calibration_threshold",
            "model_variant": "fit_only_optuna_200tree_RF_stageC_fit_subset",
            "optuna_source_parent_run_id": OPTUNA_PARENT_RUN_ID,
            "optuna_source_study": OPTUNA_DUAL_STUDY,
            "calibration_method": "sigmoid_logistic_regression",
            "threshold_tuned": "false",
            "test_labels_loaded": "false",
            "validation_used_to_fit_calibrator": "false",
            "stage_b_parent_run_id": STAGE_B_RUN_ID,
            "stage_c_split_audit_run_id": STAGE_C_AUDIT_RUN_ID,
            "stage_c_seed_selection_run_id": STAGE_C_SELECTION_RUN_ID,
            "git_commit": commit,
            "git_dirty": dirty,
        })
        mlflow.log_params({
            "dataset_sha256": pipeline.data.dataset_sha256,
            "manifest_sha256": pipeline.data.manifest_sha256,
            "curation_summary_sha256": _file_sha256(args.summary),
            "assignment_sha256": split.assignment_sha256,
            "split_csv_sha256": _file_sha256(args.split_manifest),
            "stage_c_split_audit_sha256": _file_sha256(args.split_audit),
            "training_feature_sha256": bank.fingerprint_sha256,
            "validation_feature_sha256": validation_bank.fingerprint_sha256,
            "seed": split.seed_used,
            "rf_seed": config.seed,
            "rf_trees": config.n_estimators,
            "rf_n_jobs": config.n_jobs,
            "rf_max_depth": config.max_depth,
            "rf_min_samples_leaf": config.min_samples_leaf,
            "rf_min_samples_split": config.min_samples_split,
            "rf_max_features": config.max_features,
            "optuna_selected_trial": OPTUNA_BEST_TRIAL,
            "optuna_source_run_id": OPTUNA_PARENT_RUN_ID,
            "optuna_study_name": OPTUNA_DUAL_STUDY,
        })
        log_dataset_identities(pipeline, split, args.dataset)
        mlflow.log_artifact(str(args.split_manifest), artifact_path="provenance")
        mlflow.log_artifact(str(args.split_audit), artifact_path="provenance")
        mlflow.log_artifact(str(args.manifest), artifact_path="provenance")
        mlflow.log_artifact(str(args.summary), artifact_path="provenance")
        mlflow.log_dict(asdict(config), "provenance/optimized_rf_config.json")
        mlflow.log_metric("optuna_fit_only_cv_ap", OPTUNA_BEST_CV_AP)
        mlflow.log_dict({
            "stage_b_parent_run_id": STAGE_B_RUN_ID,
            "stage_c_split_audit_run_id": STAGE_C_AUDIT_RUN_ID,
            "stage_c_seed_selection_run_id": STAGE_C_SELECTION_RUN_ID,
            "optuna_source_parent_run_id": OPTUNA_PARENT_RUN_ID,
            "optuna_study_name": OPTUNA_DUAL_STUDY,
            "optuna_selected_trial": OPTUNA_BEST_TRIAL,
            "optuna_selected_cv_ap": OPTUNA_BEST_CV_AP,
            "optuna_selection_used_fit_labels_only": True,
            "frozen_assignment_sha256": FROZEN_ASSIGNMENT_SHA256,
            "calibration_audit": audit["audits"],
            "calibration_probability_metrics_are_in_sample_for_sigmoid": True,
            "development_validation_probabilities_are_held_out": True,
            "threshold_role_used_for_fitting_calibration_or_selection": False,
            "test_partition_labels_loaded": False,
            "old_full_train_optuna_hyperparameters_reused": False,
            "split_seed_selected_after_label_balance_inspection": True,
            "development_validation_previously_explored": True,
            "stage_b_forest_reused": False,
            "uncertainty_caveat": (
                "Scaffold disjointness and in-domain calibration do not guarantee "
                "calibration on unseen scaffold distributions."
            ),
        }, "provenance/stage_d1_protocol.json")

        with mlflow.start_run(run_name="Optimized_RF_200tree_Sigmoid", nested=True):
            mlflow.set_tags({
                "task": "dual",
                "model_variant": "fit_only_optuna_200tree_stageC_fit_subset",
                "validation_only_evaluation": "true",
            })
            model = fit_sigmoid_dual_calibration(
                pipeline, split, bank, config,
                expected_assignment_sha256=FROZEN_ASSIGNMENT_SHA256,
            )
            validation = evaluate_development_validation(
                model, pipeline, validation_bank
            )
            mlflow.log_metrics({
                **_metric_dictionary(model.calibration_scores, "calibration"),
                **_metric_dictionary(validation, "validation"),
                "fit_paired_labels": float(model.n_fit_labels),
                "calibration_paired_labels": float(model.n_calibration_labels),
                "threshold_paired_labels_reserved": float(split.threshold_audit.paired_labels),
            })
            log_model_and_predictions(model, validation)
            log_comparison_figures(validation)
            mlflow.log_dict({
                "calibration_raw": asdict(model.calibration_scores.raw_metrics),
                "calibration_calibrated": asdict(model.calibration_scores.calibrated_metrics),
                "validation_raw": asdict(validation.raw_metrics),
                "validation_calibrated": asdict(validation.calibrated_metrics),
            }, "comparison/probability_metrics.json")

        mlflow.log_metrics({
            **_metric_dictionary(validation, "validation"),
            "fit_paired_labels": float(model.n_fit_labels),
            "calibration_paired_labels": float(model.n_calibration_labels),
        })
        print("\nDevelopment validation (independent of fitting and sigmoid calibration):")
        print(f"{'Metric':<20} {'Uncalibrated':>14} {'Calibrated':>14}")
        for name in ("average_precision", "brier_score", "log_loss"):
            print(
                f"{name:<20} {getattr(validation.raw_metrics, name):>14.4f} "
                f"{getattr(validation.calibrated_metrics, name):>14.4f}"
            )
        print(f"\nStage D.1 parent MLflow run: {parent.info.run_id}")
        print("Decision thresholds were not tuned; final test labels were not loaded.")
        print("Do not compare these Stage C subset-fit metrics as if Stage B used the same fit size.")


if __name__ == "__main__":
    main()
