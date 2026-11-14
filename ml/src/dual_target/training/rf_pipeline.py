"""Compose selected Morgan RF models and scaffold-held-out development evaluation.

Stage B fits selected hyperparameters on the full M2.1 *training* partition,
then scores the existing development-validation partition. The frozen test
partition is not exposed. This module does no Optuna search, calibration,
threshold tuning, plotting, or MLflow logging.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import average_precision_score

from ..data.modeling_dataset import MoleculeRecord
from ..data.pipeline import (
    PIC50_ACTIVITY_THRESHOLD,
    ModelingPipeline,
    Task,
    select_task_records,
)
from ..features.morgan_pipeline import MorganFeatureBank, build_morgan_feature_bank
from ..models.random_forest import (
    DualActivityExamples,
    RFConfig,
    TargetExamples,
    evaluate_dual_activity_classifier,
    evaluate_random_forest,
    fit_dual_activity_classifier,
    fit_random_forest,
)

type Examples = TargetExamples | DualActivityExamples
type Forest = RandomForestRegressor | RandomForestClassifier
_TASKS: tuple[Task, ...] = ("egfr", "her2", "dual")


@dataclass(frozen=True)
class RFTaskFit:
    """One fitted model, with training and validation diagnostics."""

    task: Task
    config: RFConfig
    model: Forest
    train_examples: Examples
    validation_examples: Examples
    train_metrics: dict[str, float | int | None]
    validation_metrics: dict[str, float | int | None]
    train_predictions: np.ndarray
    validation_predictions: np.ndarray


@dataclass(frozen=True)
class PairedValidation:
    """Aligned dual-activity predictions on the paired validation subset."""

    molecule_ids: tuple[str, ...]
    scaffold_ids: tuple[str, ...]
    observed_dual_active: np.ndarray
    predicted_egfr_pic50: np.ndarray
    predicted_her2_pic50: np.ndarray
    regression_ranking_score: np.ndarray
    classifier_score: np.ndarray
    regression_average_precision: float
    classifier_average_precision: float


@dataclass(frozen=True)
class RFStageBResult:
    """Three RF fits and paired evaluation, ready for an MLflow adapter."""

    egfr: RFTaskFit
    her2: RFTaskFit
    dual: RFTaskFit
    paired_validation: PairedValidation
    dataset_sha256: str
    manifest_sha256: str
    train_feature_sha256: str
    validation_feature_sha256: str


def _make_examples(
    records: Sequence[MoleculeRecord],
    task: Task,
    bank: MorganFeatureBank,
) -> Examples:
    """Compose shared task eligibility with cached molecular features."""
    selected = select_task_records(records, task)
    if not selected:
        raise ValueError(f"No accepted labels for task {task}")

    ids = tuple(record.structure_id for record in selected)
    scaffolds = tuple(record.scaffold_group_id for record in selected)
    features = bank.select_features(selected)

    if task == "dual":
        egfr = np.asarray([r.egfr_pic50 for r in selected], dtype=float)
        her2 = np.asarray([r.her2_pic50 for r in selected], dtype=float)
        if not (np.isfinite(egfr).all() and np.isfinite(her2).all()):
            raise ValueError("Paired pIC50 labels must be finite")
        labels = (
            (egfr > PIC50_ACTIVITY_THRESHOLD)
            & (her2 > PIC50_ACTIVITY_THRESHOLD)
        ).astype(np.int8)
        return DualActivityExamples(
            molecule_ids=ids,
            scaffold_ids=scaffolds,
            features=features,
            dual_active=labels,
        )

    potency = np.asarray(
        [getattr(record, f"{task}_pic50") for record in selected],
        dtype=float,
    )
    if not np.isfinite(potency).all():
        raise ValueError(f"Accepted {task} pIC50 labels must be finite")
    return TargetExamples(
        molecule_ids=ids,
        scaffold_ids=scaffolds,
        features=features,
        pic50=potency,
    )


def _fit_task(
    pipeline: ModelingPipeline,
    train_bank: MorganFeatureBank,
    validation_bank: MorganFeatureBank,
    task: Task,
    config: RFConfig,
) -> RFTaskFit:
    train = _make_examples(pipeline.data.train, task, train_bank)
    validation = _make_examples(pipeline.data.validation, task, validation_bank)

    if task == "dual":
        if not isinstance(train, DualActivityExamples) or not isinstance(
            validation, DualActivityExamples
        ):
            raise TypeError("Expected paired classification examples")
        model = fit_dual_activity_classifier(train, config)
        train_metrics, train_scores = evaluate_dual_activity_classifier(model, train)
        val_metrics, val_scores = evaluate_dual_activity_classifier(model, validation)
    else:
        if not isinstance(train, TargetExamples) or not isinstance(
            validation, TargetExamples
        ):
            raise TypeError("Expected potency regression examples")
        model = fit_random_forest(train, config)
        train_metrics, train_scores = evaluate_random_forest(model, train)
        val_metrics, val_scores = evaluate_random_forest(model, validation)

    return RFTaskFit(
        task=task,
        config=config,
        model=model,
        train_examples=train,
        validation_examples=validation,
        train_metrics=train_metrics,
        validation_metrics=val_metrics,
        train_predictions=train_scores,
        validation_predictions=val_scores,
    )


def fit_and_evaluate_selected_rf(
    pipeline: ModelingPipeline,
    train_bank: MorganFeatureBank,
    configs: Mapping[str, RFConfig],
) -> RFStageBResult:
    """Fit all three selected RFs, then evaluate on development validation.

    `configs` must contain exactly 'egfr', 'her2', and 'dual', with settings
    selected *before* development-validation evaluation. The output contains
    uncalibrated classification scores; it must not be used to fit calibrators
    or select thresholds on the validation partition.
    """
    if set(configs) != set(_TASKS):
        raise ValueError("configs must contain exactly egfr, her2, and dual")

    if train_bank.molecule_ids != tuple(
        record.structure_id for record in pipeline.data.train
    ):
        raise ValueError("Training feature bank does not match frozen train rows")

    train_ids = {record.structure_id for record in pipeline.data.train}
    validation_ids = {record.structure_id for record in pipeline.data.validation}
    if train_ids & validation_ids:
        raise ValueError("Training/validation molecule overlap")
    train_scaffolds = {
        record.scaffold_group_id for record in pipeline.data.train
    }
    validation_scaffolds = {
        record.scaffold_group_id for record in pipeline.data.validation
    }
    if train_scaffolds & validation_scaffolds:
        raise ValueError("Training/validation scaffold overlap")

    for task in _TASKS:
        config = configs[task]
        if (
            config.radius != train_bank.radius
            or config.n_bits != train_bank.n_bits
            or config.include_chirality != train_bank.include_chirality
        ):
            raise ValueError(f"{task} fingerprint configuration mismatch")

    validation_bank = build_morgan_feature_bank(
        pipeline.data.validation,
        radius=train_bank.radius,
        n_bits=train_bank.n_bits,
        include_chirality=train_bank.include_chirality,
    )
    fitted = {
        task: _fit_task(pipeline, train_bank, validation_bank, task, configs[task])
        for task in _TASKS
    }

    paired = _make_examples(pipeline.data.validation, "dual", validation_bank)
    if not isinstance(paired, DualActivityExamples):
        raise TypeError("Expected paired validation examples")

    egfr_scores = fitted["egfr"].model.predict(paired.features)
    her2_scores = fitted["her2"].model.predict(paired.features)
    regression_score = np.minimum(egfr_scores, her2_scores)

    classifier = fitted["dual"].model
    if not isinstance(classifier, RandomForestClassifier):
        raise TypeError("Expected fitted Random Forest classifier")
    positive_col = list(classifier.classes_).index(1)
    classifier_score = classifier.predict_proba(paired.features)[:, positive_col]

    paired_eval = PairedValidation(
        molecule_ids=paired.molecule_ids,
        scaffold_ids=paired.scaffold_ids,
        observed_dual_active=paired.dual_active,
        predicted_egfr_pic50=egfr_scores,
        predicted_her2_pic50=her2_scores,
        regression_ranking_score=regression_score,
        classifier_score=classifier_score,
        regression_average_precision=float(
            average_precision_score(paired.dual_active, regression_score)
        ),
        classifier_average_precision=float(
            average_precision_score(paired.dual_active, classifier_score)
        ),
    )
    return RFStageBResult(
        egfr=fitted["egfr"],
        her2=fitted["her2"],
        dual=fitted["dual"],
        paired_validation=paired_eval,
        dataset_sha256=pipeline.data.dataset_sha256,
        manifest_sha256=pipeline.data.manifest_sha256,
        train_feature_sha256=train_bank.fingerprint_sha256,
        validation_feature_sha256=validation_bank.fingerprint_sha256,
    )
