"""Stage D.2: select a raw optimized-RF dual-activity decision threshold.

Reuse the EXACT Stage D.1 Optuna-selected forest from MLflow without retraining.
Select an F1-maximizing cutoff using ONLY the frozen seed-719 threshold role
(164 paired labels); apply that frozen cutoff to M2.1 development validation
(135 paired labels) without retuning. Calibration labels reproduce the saved
D.1 predictions but are not used to choose the threshold.

Dual-active means EGFR pIC50 > 6 AND HER2 pIC50 > 6. The locked test
partition is never accessed. Historical development-validation exploration
and label-informed seed-719 selection remain methodological limitations.

Run from the repository root:
  uv run --locked python scripts/tune_dual_threshold.py \
    --run-name M2_2_StageD2_Optimized_Dual_RF_Threshold
"""

from __future__ import annotations

import argparse
import hashlib
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
    SigmoidDualCalibration,
    _make_paired_examples,
    _scores_from_probabilities,
)
from dual_target.calibration.scaffold_calibration import make_scaffold_calibration_split
from dual_target.calibration.threshold import (
    DualThresholdSelection,
    ThresholdPredictions,
    evaluate_fixed_threshold,
    evaluate_frozen_dual_threshold,
    select_dual_threshold,
)
from dual_target.data.modeling_dataset import sha256_file
from dual_target.data.pipeline import load_modeling_pipeline
from dual_target.features.morgan_pipeline import build_morgan_feature_bank
from dual_target.models.random_forest import RFConfig
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

DEFAULT_DATASET = Path("data/interim/M2_1_scaffold_split_molecules.csv")
DEFAULT_MANIFEST = Path("data/interim/M2_1_scaffold_split_manifest.csv")
DEFAULT_SUMMARY = Path("reports/audits/M2_1_scaffold_split_summary.json")
DEFAULT_SPLIT_AUDIT = Path(
    "reports/calibration/M2_2_C3_scaffold_split_2b99214bce37.json"
)
DEFAULT_SPLIT_CSV = Path(
    "reports/calibration/M2_2_C3_scaffold_split_2b99214bce37.csv"
)
DEFAULT_MLFLOW_DB = Path("reports/mlflow/mlflow.db")
FROZEN_ASSIGNMENT_SHA256 = (
    "2b99214bce37d30e1eed8f6e22984242492560b2999bb174a8d378604de9171c"
)
STAGE_B_RUN_ID = "12c9fc400b354f6da4e993e69e36b1cb"
STAGE_C_AUDIT_RUN_ID = "d2ca6b7898f34890a756c547c706bc56"
STAGE_D1_SOURCE_RUN_ID = "7f604eb2e3514ed9802d6d22fa304fb2"
OPTUNA_SOURCE_PARENT_RUN_ID = "2c748d4495ec47a4924d66043349709b"
OPTUNA_DUAL_STUDY = "M2_2_fit_only_reffirst_200trees_dual_200trees_5folds"
OPTUNA_SELECTED_TRIAL = 18
SOURCE_CHILD_NAME = "Optimized_RF_200tree_Sigmoid"
SOURCE_FOREST_PATH = "models/stage_d1_optimized_dual_rf.joblib"
SOURCE_SIGMOID_PATH = "models/stage_d1_sigmoid.joblib"
SOURCE_CALIBRATION_PATH = (
    "predictions/calibration/stage_d1_calibration_predictions.csv"
)
SOURCE_CONFIG_PATH = "provenance/optimized_rf_config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--split-audit", type=Path, default=DEFAULT_SPLIT_AUDIT)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--stage-d1-run-id", default=STAGE_D1_SOURCE_RUN_ID)
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--experiment", default="EGFR_HER2_M2_Scaffold_Benchmark")
    parser.add_argument("--run-name", default="M2_2_StageD2_Optimized_Dual_RF_Threshold")
    return parser.parse_args()


def configure_mlflow(args: argparse.Namespace) -> None:
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    elif not os.environ.get("MLFLOW_TRACKING_URI"):
        db = DEFAULT_MLFLOW_DB.resolve()
        db.parent.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(f"sqlite:///{db.as_posix()}")
    mlflow.set_experiment(args.experiment)


def git_state() -> tuple[str, str]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        changes = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
        return commit, str(bool(changes))
    except (OSError, subprocess.CalledProcessError):
        return "unknown", "unknown"


def verify_split(pipeline, split, args: argparse.Namespace) -> dict:
    """Require saved C.3 audit and manifest to identify this exact assignment."""
    if split.seed_used != 719 or split.assignment_sha256 != FROZEN_ASSIGNMENT_SHA256:
        raise ValueError("Unexpected Stage C assignment (expected frozen seed 719)")
    if (
        len(split.fit), len(split.calibration), len(split.threshold)
    ) != (5364, 1658, 1670):
        raise ValueError("Stage C molecule counts have changed")
    if (
        split.fit_audit.paired_labels,
        split.calibration_audit.paired_labels,
        split.threshold_audit.paired_labels,
    ) != (697, 223, 164):
        raise ValueError("Stage C paired-label counts have changed")
    audit = json.loads(args.split_audit.read_text(encoding="utf-8"))
    recorded = audit.get("split", {})
    source = audit.get("source", {})
    if (
        recorded.get("assignment_sha256") != split.assignment_sha256
        or recorded.get("seed_used") != 719
        or recorded.get("requested_calibration_fraction_of_scaffolds") != 0.20
        or recorded.get("requested_threshold_fraction_of_scaffolds") != 0.20
        or recorded.get("max_attempts") != 1
        or source.get("dataset_sha256") != pipeline.data.dataset_sha256
        or source.get("manifest_sha256") != pipeline.data.manifest_sha256
        or source.get("curation_summary_sha256") != sha256_file(args.summary)
    ):
        raise ValueError("Stage C audit provenance is inconsistent")
    saved_csv = args.split_manifest.read_bytes()
    if hashlib.sha256(saved_csv).hexdigest() != recorded.get("manifest_csv_sha256"):
        raise ValueError("Stage C split CSV checksum differs from audit")
    # Verify exact membership, not only a checksum asserted by an audit file.
    import csv
    import io

    rows = list(csv.DictReader(io.StringIO(saved_csv.decode("utf-8-sig"))))
    if rows != list(split.manifest_rows()):
        raise ValueError("Saved Stage C split roles differ from reconstruction")
    for role in ("fit", "calibration", "threshold"):
        recorded_role = next(
            (r for r in audit.get("audits", []) if r.get("role") == role), None
        )
        if recorded_role is None:
            raise ValueError(f"Missing Stage C audit role: {role}")
        actual_role = asdict(getattr(split, f"{role}_audit"))
        if any(recorded_role.get(k) != v for k, v in actual_role.items()):
            raise ValueError(f"Stage C {role} audit counts mismatch")
    return audit


def find_source_child(client: MlflowClient, parent_id: str):
    """Locate the unique saved optimized dual RF child within Stage D.1."""
    parent = client.get_run(parent_id)
    child_runs = client.search_runs(
        experiment_ids=[parent.info.experiment_id],
        filter_string=f"tags.`mlflow.parentRunId` = '{parent_id}'",
        max_results=1000,
    )
    matching = [
        run for run in child_runs
        if run.data.tags.get("mlflow.runName") == SOURCE_CHILD_NAME
        and run.data.tags.get("mlflow.parentRunId") == parent_id
    ]
    if len(matching) != 1:
        raise ValueError(
            f"Expected one Stage D.1 child named {SOURCE_CHILD_NAME!r}; "
            f"found {len(matching)}"
        )
    return parent, matching[0]


def load_stage_d1_model(
    client: MlflowClient, parent, child, pipeline, split, bank, directory: Path
) -> tuple[SigmoidDualCalibration, dict[str, str]]:
    """Load trusted D.1 model; reproduce its frozen calibration predictions.

    Joblib files can execute code. Load only the trusted D.1 MLflow run.
    """
    parent_params = parent.data.params
    parent_tags = parent.data.tags
    if parent_tags.get("milestone") != "M2.2_stage_D1":
        raise ValueError("Expected a Stage D.1 source MLflow run")
    if parent_tags.get("optuna_source_parent_run_id") != OPTUNA_SOURCE_PARENT_RUN_ID:
        raise ValueError("Source model is not linked to the corrected Optuna run")
    expected = {
        "dataset_sha256": pipeline.data.dataset_sha256,
        "manifest_sha256": pipeline.data.manifest_sha256,
        "assignment_sha256": split.assignment_sha256,
        "training_feature_sha256": bank.fingerprint_sha256,
        "seed": str(split.seed_used),
        "rf_seed": "714",
        "rf_trees": "200",
        "rf_max_depth": "20",
        "rf_min_samples_leaf": "2",
        "rf_min_samples_split": "4",
        "rf_max_features": "sqrt",
        "optuna_selected_trial": str(OPTUNA_SELECTED_TRIAL),
        "optuna_source_run_id": OPTUNA_SOURCE_PARENT_RUN_ID,
        "optuna_study_name": OPTUNA_DUAL_STUDY,
    }
    for key, value in expected.items():
        if parent_params.get(key) != value:
            raise ValueError(f"Stage D.1 source run differs: {key}")

    config_path = Path(client.download_artifacts(parent.info.run_id, SOURCE_CONFIG_PATH, str(directory)))
    config = RFConfig(**json.loads(config_path.read_text(encoding="utf-8")))
    expected_config = RFConfig(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=2,
        min_samples_split=4,
        max_features="sqrt",
        seed=714,
        n_jobs=config.n_jobs,
    )
    if config != expected_config:
        raise ValueError("Stage D.1 RF configuration differs from Optuna Trial 18")
    if config.n_jobs == 0 or (
        config.radius, config.n_bits, config.include_chirality
    ) != (bank.radius, bank.n_bits, bank.include_chirality):
        raise ValueError("Stage D.1 fingerprint or worker config mismatch")

    forest_path = Path(client.download_artifacts(child.info.run_id, SOURCE_FOREST_PATH, str(directory)))
    sigmoid_path = Path(client.download_artifacts(child.info.run_id, SOURCE_SIGMOID_PATH, str(directory)))
    calibration_path = Path(
        client.download_artifacts(child.info.run_id, SOURCE_CALIBRATION_PATH, str(directory))
    )
    forest = joblib.load(forest_path)
    sigmoid = joblib.load(sigmoid_path)
    if not isinstance(forest, RandomForestClassifier) or not isinstance(
        sigmoid, LogisticRegression
    ):
        raise TypeError("Unexpected fitted model types in Stage D.1 artifacts")
    if list(forest.classes_) != [0, 1] or list(sigmoid.classes_) != [0, 1]:
        raise ValueError("Stage D.1 models must contain both binary classes")
    params = forest.get_params()
    if any(
        params[name] != expected_value
        for name, expected_value in {
            "n_estimators": config.n_estimators,
            "max_depth": config.max_depth,
            "min_samples_leaf": config.min_samples_leaf,
            "min_samples_split": config.min_samples_split,
            "max_features": config.max_features,
            "random_state": config.seed,
        }.items()
    ):
        raise ValueError("Saved Stage D.1 forest parameters differ from config")

    calibration = _make_paired_examples(split.calibration, bank)
    raw = forest.predict_proba(calibration.features)[:, 1]
    calibrated = sigmoid.predict_proba(raw.reshape(-1, 1))[:, 1]
    original_csv = pd.read_csv(calibration_path, keep_default_na=False)
    required_columns = (
        "molecule_id", "scaffold_id", "observed_dual_active",
        "raw_probability", "calibrated_probability",
    )
    if any(c not in original_csv.columns for c in required_columns):
        raise ValueError("Stage D.1 saved prediction CSV is missing required columns")
    if (
        tuple(original_csv["molecule_id"].astype(str)) != calibration.molecule_ids
        or tuple(original_csv["scaffold_id"].astype(str)) != calibration.scaffold_ids
        or not np.array_equal(
            original_csv["observed_dual_active"].to_numpy(),
            calibration.dual_active,
        )
        or not np.allclose(
            original_csv["raw_probability"].to_numpy(dtype=float),
            raw, rtol=1e-9, atol=1e-12,
        )
        or not np.allclose(
            original_csv["calibrated_probability"].to_numpy(dtype=float),
            calibrated, rtol=1e-9, atol=1e-12,
        )
    ):
        raise ValueError("Loaded D.1 model does not reproduce stored calibration predictions")

    calibration_scores = _scores_from_probabilities(calibration, raw, calibrated)
    model = SigmoidDualCalibration(
        forest=forest,
        sigmoid=sigmoid,
        config=config,
        assignment_sha256=split.assignment_sha256,
        dataset_sha256=split.dataset_sha256,
        manifest_sha256=split.manifest_sha256,
        feature_sha256=bank.fingerprint_sha256,
        seed_used=split.seed_used,
        n_fit_labels=split.fit_audit.paired_labels,
        n_calibration_labels=split.calibration_audit.paired_labels,
        calibration_scores=calibration_scores,
        training_molecule_ids=frozenset(r.structure_id for r in pipeline.data.train),
        training_scaffold_ids=frozenset(r.scaffold_group_id for r in pipeline.data.train),
    )
    source_hashes = {
        "source_forest_sha256": sha256_file(forest_path),
        "source_sigmoid_sha256": sha256_file(sigmoid_path),
        "source_calibration_csv_sha256": sha256_file(calibration_path),
        "source_rf_config_sha256": sha256_file(config_path),
    }
    return model, source_hashes


def prediction_frame(rows: ThresholdPredictions) -> pd.DataFrame:
    return pd.DataFrame({
        "molecule_id": rows.molecule_ids,
        "scaffold_id": rows.scaffold_ids,
        "observed_dual_active": rows.observed,
        "raw_probability": rows.scores,
        "predicted_dual_active": rows.predicted,
        "selected_threshold": rows.metrics.threshold,
    })


def curve_frame(selection: DualThresholdSelection) -> pd.DataFrame:
    return pd.DataFrame([asdict(row) for row in selection.curve])


def plot_threshold_curves(selection: DualThresholdSelection) -> plt.Figure:
    curve = curve_frame(selection).sort_values("threshold")
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for column, label in (
        ("f1", "F1"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("specificity", "Specificity"),
    ):
        ax.plot(curve["threshold"], curve[column], label=label, linewidth=1.7)
    ax.axvline(
        selection.threshold, color="black", linestyle="--",
        label=f"Selected threshold = {selection.threshold:.3f}",
    )
    ax.set(
        xlabel="Raw RF probability cutoff (score ≥ threshold)",
        ylabel="Score",
        title="Frozen Stage C threshold role — F1 selected without validation labels",
        xlim=(0, 1), ylim=(0, 1.05),
    )
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_confusion_matrices(
    threshold: ThresholdPredictions, validation: ThresholdPredictions
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.3))
    for ax, rows, title in (
        (axes[0], threshold, "Selection subset (n=164)"),
        (axes[1], validation, "Development validation (n=135)"),
    ):
        m = rows.metrics
        matrix = np.asarray([[m.tn, m.fp], [m.fn, m.tp]], dtype=int)
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, int(matrix.max())))
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=13)
        ax.set(
            title=f"{title}\nF1={m.f1:.3f}",
            xlabel="Predicted class", ylabel="Observed class",
            xticks=[0, 1], yticks=[0, 1],
        )
        ax.set_xticklabels(["Not dual", "Dual"])
        ax.set_yticklabels(["Not dual", "Dual"])
    fig.suptitle("Frozen cutoff applied without validation retuning")
    fig.tight_layout()
    return fig


def plot_score_distributions(
    threshold: ThresholdPredictions, validation: ThresholdPredictions
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), sharey=True)
    bins = np.linspace(0, 1, 18)
    for ax, rows, title in (
        (axes[0], threshold, "Threshold-selection subset"),
        (axes[1], validation, "Development validation"),
    ):
        for label, name in ((0, "Not dual-active"), (1, "Dual-active")):
            ax.hist(
                rows.scores[rows.observed == label], bins=bins,
                histtype="step", linewidth=1.7, label=name,
            )
        ax.axvline(threshold.metrics.threshold, color="black", ls="--")
        ax.set(xlim=(0, 1), xlabel="Raw RF positive-class probability", title=title)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Molecules")
    fig.suptitle("Class-conditional scores with frozen F1 threshold")
    fig.tight_layout()
    return fig


def log_plot(fig: plt.Figure, filename: str) -> None:
    try:
        mlflow.log_figure(fig, f"plots/{filename}")
    finally:
        plt.close(fig)


def prefix_metrics(rows: ThresholdPredictions, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_{key}": float(value)
        for key, value in asdict(rows.metrics).items()
    }


def log_outputs(
    selection: DualThresholdSelection,
    validation: ThresholdPredictions,
    source_hashes: dict[str, str],
    source_d1_id: str,
    source_child_id: str,
) -> None:
    """Log complete, reproducible evidence within the current MLflow run."""
    threshold = selection.threshold_predictions
    baseline_t = evaluate_fixed_threshold(threshold.observed, threshold.scores, 0.5)
    baseline_v = evaluate_fixed_threshold(validation.observed, validation.scores, 0.5)
    mlflow.log_metrics({
        **prefix_metrics(threshold, "selection"),
        **prefix_metrics(validation, "validation"),
        "threshold_default_0_5_selection_f1": baseline_t.f1,
        "threshold_default_0_5_validation_f1": baseline_v.f1,
        "threshold_default_0_5_validation_precision": baseline_v.precision,
        "threshold_default_0_5_validation_recall": baseline_v.recall,
    })
    mlflow.log_params({
        "objective": selection.objective,
        "score_kind": selection.score_kind,
        "selected_threshold": selection.threshold,
        "seed_used": selection.seed_used,
        "assignment_sha256": selection.assignment_sha256,
        "dataset_sha256": selection.dataset_sha256,
        "manifest_sha256": selection.manifest_sha256,
        "train_feature_sha256": selection.train_feature_sha256,
        "n_fit_labels": selection.n_fit_labels,
        "n_calibration_labels": selection.n_calibration_labels,
        "n_threshold_labels": threshold.metrics.n,
        "n_validation_labels": validation.metrics.n,
    })
    mlflow.log_dict({
        "decision_rule": "predict dual-active when raw RF score >= selected_threshold",
        "activity_definition": "EGFR pIC50 > 6 AND HER2 pIC50 > 6",
        "objective": selection.objective,
        "tie_break": "highest score cutoff on equal F1",
        "score_kind": selection.score_kind,
        "selected_threshold": selection.threshold,
        "selected_threshold_metrics": asdict(selection.selected_metrics),
        "validation_metrics": asdict(validation.metrics),
        "default_0_5_threshold_metrics": asdict(baseline_t),
        "default_0_5_validation_metrics": asdict(baseline_v),
        "d1_source_parent_run": source_d1_id,
        "d1_source_child_run": source_child_id,
        "optuna_parent_run": OPTUNA_SOURCE_PARENT_RUN_ID,
        "optuna_study": OPTUNA_DUAL_STUDY,
        "optuna_selected_trial": OPTUNA_SELECTED_TRIAL,
        "source_artifact_hashes": source_hashes,
        "protocol_notes": [
            "Threshold selected using the 164 paired Stage C threshold-role labels only.",
            "Development-validation labels used only after threshold frozen.",
            "No RF or sigmoid refitting in Stage D.2; exact D.1 forest reloaded.",
            (
                "Raw scores favored after reviewing Stage D.1 development-validation results; "
                "hence final model-selection performance is exploratory."
            ),
            "Stage C seed 719 was selected after observing label balance.",
            "F1 is prevalence-dependent and may not align with screening costs.",
            "Final M2.1 test labels were not accessed.",
        ],
    }, "decision/selected_rule.json")
    with TemporaryDirectory() as dirname:
        output = Path(dirname)
        c = output / "threshold_sweep.csv"
        t = output / "threshold_predictions.csv"
        v = output / "validation_predictions.csv"
        curve_frame(selection).to_csv(c, index=False)
        prediction_frame(threshold).to_csv(t, index=False)
        prediction_frame(validation).to_csv(v, index=False)
        mlflow.log_artifact(str(c), artifact_path="curves")
        mlflow.log_artifact(str(t), artifact_path="predictions")
        mlflow.log_artifact(str(v), artifact_path="predictions")
    log_plot(plot_threshold_curves(selection), "selection_threshold_curves.png")
    log_plot(
        plot_confusion_matrices(threshold, validation),
        "selection_vs_validation_confusion.png",
    )
    log_plot(
        plot_score_distributions(threshold, validation),
        "selection_validation_score_distributions.png",
    )


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
    audit = verify_split(pipeline, split, args)
    train_bank = build_morgan_feature_bank(pipeline.data.train)
    validation_bank = build_morgan_feature_bank(pipeline.data.validation)

    configure_mlflow(args)
    client = MlflowClient()
    source_parent, source_child = find_source_child(client, args.stage_d1_run_id)
    with TemporaryDirectory() as folder:
        model, source_hashes = load_stage_d1_model(
            client, source_parent, source_child, pipeline, split,
            train_bank, Path(folder),
        )
        if (
            validation_bank.radius,
            validation_bank.n_bits,
            validation_bank.include_chirality,
        ) != (
            model.config.radius,
            model.config.n_bits,
            model.config.include_chirality,
        ):
            raise ValueError("Stage C validation fingerprint config mismatch")

        # Both calculations use tested, pure threshold utilities. Neither can
        # change the source forest, its sigmoid, or the frozen original labels.
        selection = select_dual_threshold(
            model, pipeline, split, train_bank,
            expected_assignment_sha256=FROZEN_ASSIGNMENT_SHA256,
            score_kind="raw",
        )
        validation = evaluate_frozen_dual_threshold(
            selection, model, pipeline, validation_bank
        )

        print("STAGE D.2 — OPTIMIZED DUAL RF DECISION-THRESHOLD SELECTION")
        print(f"Assignment SHA-256: {split.assignment_sha256}")
        print(f"Stage D.1 source MLflow parent: {source_parent.info.run_id}")
        print(f"Stage D.1 source MLflow child: {source_child.info.run_id}")
        print(f"Original training partition: {len(pipeline.data.train)} molecules")
        print(f"RF fitting subset: {len(split.fit)} molecules")
        print(f"RF fitting paired labels: {split.fit_audit.paired_labels}")
        print(f"Calibration paired labels: {split.calibration_audit.paired_labels}")
        print(f"Threshold-selection paired labels: {split.threshold_audit.paired_labels}")
        print(f"Paired fit={split.fit_audit.paired_labels}; "
              f"calibration={split.calibration_audit.paired_labels}; "
              f"threshold selection={split.threshold_audit.paired_labels}; "
              f"development validation={validation.metrics.n}")
        print(f"Selected raw-RF cutoff (F1): {selection.threshold:.6f}")
        print(f"Selection: F1={selection.selected_metrics.f1:.4f}, "
              f"precision={selection.selected_metrics.precision:.4f}, "
              f"recall={selection.selected_metrics.recall:.4f}")
        print(f"Validation: F1={validation.metrics.f1:.4f}, "
              f"precision={validation.metrics.precision:.4f}, "
              f"recall={validation.metrics.recall:.4f}, "
              f"TP={validation.metrics.tp}, FP={validation.metrics.fp}, "
              f"TN={validation.metrics.tn}, FN={validation.metrics.fn}")

        with mlflow.start_run(run_name=args.run_name) as parent:
            commit, dirty = git_state()
            mlflow.set_tags({
                "milestone": "M2.2_stage_D2",
                "task": "dual_activity_thresholding",
                "selection_objective": "f1",
                "score_kind": "raw",
                "source_d1_parent_run_id": source_parent.info.run_id,
                "source_d1_child_run_id": source_child.info.run_id,
                "optuna_source_parent_run_id": OPTUNA_SOURCE_PARENT_RUN_ID,
                "optuna_study_name": OPTUNA_DUAL_STUDY,
                "stage_b_parent_run_id": STAGE_B_RUN_ID,
                "stage_c_split_audit_run_id": STAGE_C_AUDIT_RUN_ID,
                "validation_used_for_threshold_selection": "false",
                "forest_retrained": "false",
                "test_labels_loaded": "false",
                "git_commit": commit,
                "git_dirty": dirty,
            })
            mlflow.log_params({
                "dataset_sha256": pipeline.data.dataset_sha256,
                "manifest_sha256": pipeline.data.manifest_sha256,
                "assignment_sha256": split.assignment_sha256,
                "train_feature_sha256": train_bank.fingerprint_sha256,
                "validation_feature_sha256": validation_bank.fingerprint_sha256,
                "split_csv_sha256": sha256_file(args.split_manifest),
                "split_audit_sha256": sha256_file(args.split_audit),
                "curation_summary_sha256": sha256_file(args.summary),
            })
            for role, records in (
                ("fit", split.fit),
                ("calibration", split.calibration),
                ("threshold", split.threshold),
                ("validation", pipeline.data.validation),
            ):
                identities = pd.DataFrame({
                    "molecule_id": [r.structure_id for r in records],
                    "scaffold_id": [r.scaffold_group_id for r in records],
                })
                mlflow.log_input(
                    mlflow.data.from_pandas(
                        identities,
                        name=f"M2_1_{role}_identities_C6",
                    ),
                    context="training" if role == "fit" else role,
                )
            mlflow.log_artifact(str(args.split_manifest), artifact_path="provenance")
            mlflow.log_artifact(str(args.split_audit), artifact_path="provenance")
            mlflow.log_artifact(str(args.manifest), artifact_path="provenance")
            mlflow.log_artifact(str(args.summary), artifact_path="provenance")
            mlflow.log_dict({
                "source_parent_run": source_parent.info.run_id,
                "source_child_run": source_child.info.run_id,
                "c3_split_assignment": split.assignment_sha256,
                "c3_audit_counts": audit.get("audits"),
                "source_artifact_hashes": source_hashes,
                "model_loaded_from_mlflow_not_refitted": True,
                "optuna_parent_run": OPTUNA_SOURCE_PARENT_RUN_ID,
                "optuna_study": OPTUNA_DUAL_STUDY,
                "optuna_selected_trial": OPTUNA_SELECTED_TRIAL,
                "score_kind": "raw",
                "development_validation_previously_explored": True,
                "stage_c_seed_selected_after_label_balance_inspection": True,
            }, "provenance/stage_d2_protocol.json")

            with mlflow.start_run(
                run_name="Optimized_RF_200tree_Raw_F1_Threshold", nested=True
            ):
                mlflow.set_tags({
                    "task": "dual",
                    "source_stage": "D1_fit_only_optuna_optimized_RF",
                    "threshold_selected_on": "C3_threshold_role",
                    "threshold_applied_to": "M2_1_development_validation",
                })
                log_outputs(
                    selection, validation, source_hashes,
                    source_parent.info.run_id, source_child.info.run_id,
                )
            mlflow.log_metrics({
                "selected_threshold": selection.threshold,
                "threshold_selection_f1": selection.selected_metrics.f1,
                "validation_f1": validation.metrics.f1,
                "validation_precision": validation.metrics.precision,
                "validation_recall": validation.metrics.recall,
            })
            print(f"\nStage D.2 parent MLflow run: {parent.info.run_id}")
            print("Plots: child run / Artifacts / plots")
            print("Threshold frozen before validation. No final test labels loaded.")


if __name__ == "__main__":
    main()
