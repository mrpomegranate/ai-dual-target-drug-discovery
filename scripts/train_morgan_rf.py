"""Train EGFR/HER2 Morgan RF baselines and log MLflow diagnostics.

Only frozen M2.1 train and validation molecules are loaded. The test partition
is reserved for the final locked evaluation.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import rdkit
import sklearn
from dual_target.data.modeling_dataset import load_modeling_data, sha256_file
from dual_target.models.random_forest import (
    RFConfig,
    evaluate_dual_activity_classifier,
    evaluate_random_forest,
    fit_dual_activity_classifier,
    fit_random_forest,
    make_dual_activity_examples,
    make_target_examples,
)
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    precision_recall_curve,
)

DEFAULT_DATASET = Path("data/interim/M2_1_scaffold_split_molecules.csv")
DEFAULT_MANIFEST = Path("data/interim/M2_1_scaffold_split_manifest.csv")
DEFAULT_SUMMARY = Path("reports/audits/M2_1_scaffold_split_summary.json")
ACTIVITY_THRESHOLD_PIC50 = 6.0
CLASSIFICATION_THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--run-name", default="RF_Morgan_M2_2")
    parser.add_argument("--experiment", default="EGFR_HER2_M2_Scaffold_Benchmark")
    parser.add_argument("--tracking-uri", default=None)
    args = parser.parse_args()
    if args.trees < 1:
        parser.error("--trees must be positive")
    if args.n_jobs == 0:
        parser.error("--n-jobs must not be zero")
    return args


def configure_tracking(tracking_uri: str | None, experiment: str) -> None:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    elif not os.environ.get("MLFLOW_TRACKING_URI"):
        db_path = Path("reports/mlflow/mlflow.db").resolve()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")
    mlflow.set_experiment(experiment)


def git_version() -> tuple[str, str]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        changes = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return commit, str(bool(changes))
    except (OSError, subprocess.CalledProcessError):
        return "unknown", "unknown"


def log_provenance(data, config: RFConfig, family: str) -> None:
    commit, dirty = git_version()
    mlflow.set_tags(
        {
            "model_family": family,
            "evaluation_split": "validation",
            "test_labels_used": "false",
            "git_commit": commit,
            "git_dirty": dirty,
        }
    )
    params = {
        **asdict(config),
        "dataset_sha256": data.dataset_sha256,
        "manifest_sha256": data.manifest_sha256,
        "feature_type": "Morgan_bit_vector",
        "split_method": "Bemis_Murcko_scaffold",
        "python_version": platform.python_version(),
        "rdkit_version": rdkit.__version__,
        "sklearn_version": sklearn.__version__,
        "mlflow_version": mlflow.__version__,
    }
    mlflow.log_params({key: str(value) for key, value in params.items()})


def log_parent_datasets(data, source: Path) -> None:
    """Show the frozen train and validation datasets on the parent MLflow row."""
    for stage, records in (("training", data.train), ("validation", data.validation)):
        frame = pd.DataFrame(
            {
                "molecule_id": [r.structure_id for r in records],
                "scaffold_id": [r.scaffold_group_id for r in records],
                "egfr_mask": [str(r.egfr_mask) for r in records],
                "her2_mask": [str(r.her2_mask) for r in records],
            }
        )
        dataset = mlflow.data.from_pandas(
            frame,
            source=source.resolve().as_uri(),
            name=f"M2_1_all_targets_{stage}",
        )
        mlflow.log_input(dataset, context=stage)


def log_target_dataset(examples, name: str, stage: str, source: Path) -> None:
    values = examples.pic50 if hasattr(examples, "pic50") else examples.dual_active
    frame = pd.DataFrame(
        {
            "molecule_id": examples.molecule_ids,
            "scaffold_id": examples.scaffold_ids,
            "label": np.asarray(values, dtype=np.float64),
        }
    )
    dataset = mlflow.data.from_pandas(
        frame,
        source=source.resolve().as_uri(),
        name=f"M2_1_{name}_{stage}",
        targets="label",
    )
    mlflow.log_input(dataset, context=stage)


def log_provenance_artifacts(data, args: argparse.Namespace) -> None:
    """Preserve split evidence, but do not upload the entire modeling CSV."""
    mlflow.log_artifact(str(args.manifest), artifact_path="provenance")
    mlflow.log_artifact(str(args.summary), artifact_path="provenance")
    mlflow.log_dict(
        {
            "dataset_path": str(args.dataset.resolve()),
            "dataset_sha256": data.dataset_sha256,
            "manifest_path": str(args.manifest.resolve()),
            "manifest_sha256": data.manifest_sha256,
            "summary_sha256": sha256_file(args.summary),
            "test_labels_used": False,
        },
        "provenance/checksums.json",
    )


def log_model_and_predictions(
    model,
    train,
    train_predictions: np.ndarray,
    validation,
    validation_predictions: np.ndarray,
    label_name: str,
) -> None:
    """Save the model and molecule-aligned predictions for both observed splits."""
    with TemporaryDirectory() as temporary:
        folder = Path(temporary)
        model_path = folder / "model.joblib"
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="models")

        for stage, examples, predictions in (
            ("training", train, train_predictions),
            ("validation", validation, validation_predictions),
        ):
            path = folder / f"{stage}_predictions.csv"
            pd.DataFrame(
                {
                    "molecule_id": examples.molecule_ids,
                    "scaffold_id": examples.scaffold_ids,
                    "observed": getattr(examples, label_name),
                    "predicted": predictions,
                }
            ).to_csv(path, index=False)
            mlflow.log_artifact(str(path), artifact_path=f"predictions/{stage}")


def log_plot(fig: plt.Figure, filename: str) -> None:
    """Save one Matplotlib PNG directly in the active run's artifacts."""
    try:
        fig.tight_layout()
        mlflow.log_figure(fig, f"plots/{filename}")
    finally:
        plt.close(fig)


def plot_regression_diagnostics(
    target: str,
    train,
    train_predictions: np.ndarray,
    train_metrics: dict,
    validation,
    validation_predictions: np.ndarray,
    validation_metrics: dict,
) -> None:
    name = target.upper()
    actual_train = train.pic50
    actual_val = validation.pic50
    train_residual = train_predictions - actual_train
    val_residual = validation_predictions - actual_val

    # Parity: predictions against measured potency (ideal predictions lie on y=x).
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actual_train, train_predictions, alpha=0.3, s=12, label="Train")
    ax.scatter(actual_val, validation_predictions, alpha=0.65, s=16, label="Validation")
    bounds = np.concatenate(
        [actual_train, actual_val, train_predictions, validation_predictions]
    )
    low, high = float(bounds.min()), float(bounds.max())
    ax.plot([low, high], [low, high], "--", label="Ideal prediction")
    ax.set(xlabel="Measured pIC50", ylabel="Predicted pIC50", title=f"{name}: parity")
    ax.legend()
    log_plot(fig, f"{target}_parity.png")

    # Residual bias and heteroscedasticity.
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(train_predictions, train_residual, alpha=0.3, s=12, label="Train")
    ax.scatter(
        validation_predictions, val_residual, alpha=0.65, s=16, label="Validation"
    )
    ax.axhline(0, linestyle="--")
    ax.set(
        xlabel="Predicted pIC50",
        ylabel="Residual (predicted - measured)",
        title=f"{name}: residuals",
    )
    ax.legend()
    log_plot(fig, f"{target}_residuals.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(train_residual, bins=25, density=True, alpha=0.5, label="Train")
    ax.hist(val_residual, bins=25, density=True, alpha=0.5, label="Validation")
    ax.axvline(0, linestyle="--")
    ax.set(
        xlabel="Residual (predicted - measured)",
        ylabel="Density",
        title=f"{name}: residual distribution",
    )
    ax.legend()
    log_plot(fig, f"{target}_residual_histogram.png")

    fig, ax = plt.subplots(figsize=(5, 4))
    values = [float(train_metrics["mae"]), float(validation_metrics["mae"])]
    bars = ax.bar(["Train", "Validation"], values)
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set(ylabel="MAE (pIC50 units)", title=f"{name}: train vs validation MAE")
    ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1)
    log_plot(fig, f"{target}_mae_comparison.png")


def plot_classifier_diagnostics(
    train,
    train_probabilities: np.ndarray,
    train_metrics: dict,
    validation,
    validation_probabilities: np.ndarray,
    validation_metrics: dict,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(
        validation.dual_active,
        validation_probabilities,
        ax=ax,
        name="Direct RF classifier",
    )
    ax.axhline(
        float(validation_metrics["prevalence"]), linestyle="--", label="Prevalence"
    )
    ax.set_title("Dual-active classifier: validation PR curve")
    ax.legend()
    log_plot(fig, "dual_classifier_pr_curve.png")

    predicted_class = (validation_probabilities >= CLASSIFICATION_THRESHOLD).astype(int)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        validation.dual_active,
        predicted_class,
        labels=[0, 1],
        display_labels=["Not dual-active", "Dual-active"],
        ax=ax,
        colorbar=False,
    )
    ax.set_title("Direct RF: validation confusion matrix (threshold=0.5)")
    log_plot(fig, "dual_classifier_confusion_matrix.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = validation.dual_active
    ax.hist(
        validation_probabilities[labels == 0],
        bins=20,
        alpha=0.55,
        label="Not dual-active",
    )
    ax.hist(
        validation_probabilities[labels == 1],
        bins=20,
        alpha=0.55,
        label="Dual-active",
    )
    ax.axvline(CLASSIFICATION_THRESHOLD, linestyle="--", label="0.5 threshold")
    ax.set(
        xlabel="Predicted dual-active probability",
        ylabel="Molecules",
        title="Direct RF: validation score distributions",
    )
    ax.legend()
    log_plot(fig, "dual_classifier_score_histogram.png")

    fig, ax = plt.subplots(figsize=(5, 4))
    values = [
        float(train_metrics["average_precision"]),
        float(validation_metrics["average_precision"]),
    ]
    bars = ax.bar(["Train", "Validation"], values)
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set(
        ylabel="Average precision",
        ylim=(0, 1.1),
        title="Direct RF: train vs validation AP",
    )
    log_plot(fig, "dual_classifier_ap_comparison.png")


def plot_paired_comparison(
    labels: np.ndarray,
    regression_scores: np.ndarray,
    classifier_scores: np.ndarray,
    egfr_predictions: np.ndarray,
    her2_predictions: np.ndarray,
) -> None:
    prevalence = float(labels.mean())
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, scores in (
        ("Regression-derived", regression_scores),
        ("Direct classifier", classifier_scores),
    ):
        precision, recall, _ = precision_recall_curve(labels, scores)
        ap = average_precision_score(labels, scores)
        ax.step(recall, precision, where="post", label=f"{name} (AP={ap:.3f})")
    ax.axhline(prevalence, linestyle="--", label=f"Prevalence={prevalence:.3f}")
    ax.set(
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0, 1),
        ylim=(0, 1.05),
        title="Dual-active ranking: paired validation molecules",
    )
    ax.legend()
    log_plot(fig, "dual_activity_pr_comparison.png")

    regression_classes = (
        (egfr_predictions > ACTIVITY_THRESHOLD_PIC50)
        & (her2_predictions > ACTIVITY_THRESHOLD_PIC50)
    ).astype(int)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        labels,
        regression_classes,
        labels=[0, 1],
        display_labels=["Not dual-active", "Dual-active"],
        ax=ax,
        colorbar=False,
    )
    ax.set_title("Regression-derived dual activity: validation confusion matrix")
    log_plot(fig, "dual_from_regression_confusion_matrix.png")

    fig, ax = plt.subplots(figsize=(5, 4))
    values = [
        float(average_precision_score(labels, regression_scores)),
        float(average_precision_score(labels, classifier_scores)),
    ]
    bars = ax.bar(["Regression-derived", "Direct classifier"], values)
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set(
        ylabel="Average precision", ylim=(0, 1.1), title="Dual-activity AP comparison"
    )
    log_plot(fig, "dual_activity_ap_comparison.png")


def run_regression(target: str, data, config: RFConfig, source: Path):
    train = make_target_examples(data.train, target, config)
    validation = make_target_examples(data.validation, target, config)
    with mlflow.start_run(run_name=f"RF_{target.upper()}_pIC50", nested=True):
        log_provenance(data, config, "rf_regression")
        mlflow.log_param("target", target)
        mlflow.log_param("train_labels", len(train.pic50))
        mlflow.log_param("validation_labels", len(validation.pic50))
        log_target_dataset(train, target, "training", source)
        log_target_dataset(validation, target, "validation", source)

        model = fit_random_forest(train, config)
        train_metrics, train_predictions = evaluate_random_forest(model, train)
        validation_metrics, validation_predictions = evaluate_random_forest(
            model, validation
        )
        logged_metrics = {
            **{
                f"train_{key}": float(value)
                for key, value in train_metrics.items()
                if value is not None
            },
            **{
                f"validation_{key}": float(value)
                for key, value in validation_metrics.items()
                if value is not None
            },
            "mae_generalization_gap": float(
                validation_metrics["mae"] - train_metrics["mae"]
            ),
        }
        mlflow.log_metrics(logged_metrics)
        log_model_and_predictions(
            model,
            train,
            train_predictions,
            validation,
            validation_predictions,
            "pic50",
        )
        plot_regression_diagnostics(
            target,
            train,
            train_predictions,
            train_metrics,
            validation,
            validation_predictions,
            validation_metrics,
        )
        print(f"{target.upper()} regression: {validation_metrics}")
        print(f"{target.upper()} train MAE: {train_metrics['mae']:.4f}")
    return model, train_metrics, validation_metrics


def run_classification(data, config: RFConfig, source: Path):
    train = make_dual_activity_examples(data.train, config)
    validation = make_dual_activity_examples(data.validation, config)
    with mlflow.start_run(run_name="RF_DUAL_classifier", nested=True):
        log_provenance(data, config, "rf_dual_classifier")
        mlflow.log_param("activity_threshold_pic50", ACTIVITY_THRESHOLD_PIC50)
        mlflow.log_param("classification_threshold", CLASSIFICATION_THRESHOLD)
        mlflow.log_param("train_pairs", len(train.dual_active))
        mlflow.log_param("validation_pairs", len(validation.dual_active))
        log_target_dataset(train, "dual", "training", source)
        log_target_dataset(validation, "dual", "validation", source)

        model = fit_dual_activity_classifier(train, config)
        train_metrics, train_probabilities = evaluate_dual_activity_classifier(
            model, train
        )
        validation_metrics, validation_probabilities = (
            evaluate_dual_activity_classifier(model, validation)
        )
        mlflow.log_metrics(
            {
                **{
                    f"train_{key}": float(value)
                    for key, value in train_metrics.items()
                },
                **{
                    f"validation_{key}": float(value)
                    for key, value in validation_metrics.items()
                },
                "ap_generalization_gap": float(
                    train_metrics["average_precision"]
                    - validation_metrics["average_precision"]
                ),
            }
        )
        log_model_and_predictions(
            model,
            train,
            train_probabilities,
            validation,
            validation_probabilities,
            "dual_active",
        )
        plot_classifier_diagnostics(
            train,
            train_probabilities,
            train_metrics,
            validation,
            validation_probabilities,
            validation_metrics,
        )
        print(f"Direct dual classifier: {validation_metrics}")
        print(f"Classifier train AP: {train_metrics['average_precision']:.4f}")
    return model, train_metrics, validation_metrics


def main() -> None:
    args = parse_args()
    config = RFConfig(n_estimators=args.trees, n_jobs=args.n_jobs)
    data = load_modeling_data(args.dataset, args.manifest, args.summary)
    configure_tracking(args.tracking_uri, args.experiment)

    print(f"Train molecules: {len(data.train)}; validation: {len(data.validation)}")
    print(f"Dataset SHA-256: {data.dataset_sha256}")
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

    with mlflow.start_run(run_name=args.run_name) as parent:
        log_provenance(data, config, "rf_baseline_comparison")
        log_parent_datasets(data, args.dataset)
        log_provenance_artifacts(data, args)

        egfr_model, egfr_train, egfr_val = run_regression(
            "egfr", data, config, args.dataset
        )
        her2_model, her2_train, her2_val = run_regression(
            "her2", data, config, args.dataset
        )
        classifier_model, classifier_train, _classifier_val = run_classification(
            data, config, args.dataset
        )

        paired = make_dual_activity_examples(data.validation, config)
        egfr_predictions = egfr_model.predict(paired.features)
        her2_predictions = her2_model.predict(paired.features)
        joint_score = np.minimum(egfr_predictions, her2_predictions)
        classifier_positive_column = list(classifier_model.classes_).index(1)
        classifier_score = classifier_model.predict_proba(paired.features)[
            :, classifier_positive_column
        ]
        joint_ap = float(average_precision_score(paired.dual_active, joint_score))
        classifier_ap = float(
            average_precision_score(paired.dual_active, classifier_score)
        )
        mlflow.log_metrics(
            {
                "train_egfr_mae": float(egfr_train["mae"]),
                "train_her2_mae": float(her2_train["mae"]),
                "validation_egfr_mae": float(egfr_val["mae"]),
                "validation_her2_mae": float(her2_val["mae"]),
                "train_dual_classifier_ap": float(
                    classifier_train["average_precision"]
                ),
                "validation_dual_ap_from_regression": joint_ap,
                "validation_dual_ap_from_classifier": classifier_ap,
                "validation_dual_prevalence": float(paired.dual_active.mean()),
            }
        )
        with TemporaryDirectory() as temporary:
            predictions_path = Path(temporary) / "paired_regression_scores.csv"
            pd.DataFrame(
                {
                    "molecule_id": paired.molecule_ids,
                    "dual_active": paired.dual_active,
                    "predicted_egfr_pic50": egfr_predictions,
                    "predicted_her2_pic50": her2_predictions,
                    "joint_regression_score": joint_score,
                    "classifier_dual_probability": classifier_score,
                }
            ).to_csv(predictions_path, index=False)
            mlflow.log_artifact(str(predictions_path), artifact_path="validation")

        plot_paired_comparison(
            paired.dual_active,
            joint_score,
            classifier_score,
            egfr_predictions,
            her2_predictions,
        )
        mlflow.log_dict(
            {
                "regression_score": "Minimum predicted EGFR/HER2 pIC50",
                "classifier_score": "Random Forest predicted dual-active probability",
                "regression_binary_threshold": "EGFR pIC50 > 6 AND HER2 pIC50 > 6",
                "classifier_binary_threshold": CLASSIFICATION_THRESHOLD,
                "validation_paired_molecules": len(paired.dual_active),
                "test_partition_used": False,
            },
            "evaluation/methodology.json",
        )
        print(f"Dual-activity AP, regression: {joint_ap:.4f}")
        print(f"Dual-activity AP, classifier: {classifier_ap:.4f}")
        print(f"Parent MLflow run ID: {parent.info.run_id}")


if __name__ == "__main__":
    main()
