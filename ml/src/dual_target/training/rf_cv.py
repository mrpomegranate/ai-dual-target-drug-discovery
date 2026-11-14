"""Scaffold-aware cross-validation for cached Morgan Random Forest models.

Only the frozen M2.1 training partition is used. Validation and test
partitions are never used for fitting or scoring within this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.metrics import average_precision_score, mean_absolute_error

from ..data.modeling_dataset import MoleculeRecord
from ..data.pipeline import (
    PIC50_ACTIVITY_THRESHOLD,
    ModelingPipeline,
    Task,
    select_task_records,
)
from ..features.morgan_pipeline import MorganFeatureBank
from ..models.random_forest import (
    DualActivityExamples,
    RFConfig,
    TargetExamples,
    evaluate_dual_activity_classifier,
    evaluate_random_forest,
    fit_dual_activity_classifier,
    fit_random_forest,
)


@dataclass(frozen=True)
class FoldMetrics:
    """Train/holdout scores from one internal scaffold-disjoint fold."""

    fold: int
    n_fit: int
    n_holdout: int
    train_score: float
    holdout_score: float
    holdout_spearman: float | None = None
    holdout_positives: int | None = None
    holdout_prevalence: float | None = None


@dataclass(frozen=True)
class OOFPrediction:
    """A molecule prediction generated without fitting on its scaffold."""

    structure_id: str
    scaffold_group_id: str
    fold: int
    observed: float
    predicted: float


@dataclass(frozen=True)
class RFCVResult:
    """Reusable results for Optuna and later calibration design."""

    task: Task
    metric_name: Literal["mae", "average_precision"]
    fold_metrics: tuple[FoldMetrics, ...]
    oof_predictions: tuple[OOFPrediction, ...]
    mean_holdout_score: float
    pooled_oof_score: float


def _examples(
    records: tuple[MoleculeRecord, ...],
    task: Task,
    bank: MorganFeatureBank,
) -> TargetExamples | DualActivityExamples:
    """Select accepted target labels and matching cached feature rows."""
    selected = select_task_records(records, task)
    if not selected:
        raise ValueError(f"No accepted {task} labels in fold")

    molecule_ids = tuple(record.structure_id for record in selected)
    scaffold_ids = tuple(record.scaffold_group_id for record in selected)
    features = bank.select_features(selected)

    if task == "dual":
        egfr = np.asarray([record.egfr_pic50 for record in selected], dtype=float)
        her2 = np.asarray([record.her2_pic50 for record in selected], dtype=float)
        if not (np.all(np.isfinite(egfr)) and np.all(np.isfinite(her2))):
            raise ValueError("Nonfinite paired pIC50 label")
        labels = ((egfr > PIC50_ACTIVITY_THRESHOLD) & (her2 > PIC50_ACTIVITY_THRESHOLD))
        return DualActivityExamples(
            molecule_ids=molecule_ids,
            scaffold_ids=scaffold_ids,
            features=features,
            dual_active=labels.astype(np.int8),
        )

    labels = np.asarray(
        [getattr(record, f"{task}_pic50") for record in selected],
        dtype=float,
    )
    if not np.all(np.isfinite(labels)):
        raise ValueError(f"Nonfinite {task} pIC50 label")
    return TargetExamples(
        molecule_ids=molecule_ids,
        scaffold_ids=scaffold_ids,
        features=features,
        pic50=labels,
    )


def run_rf_scaffold_cv(
    pipeline: ModelingPipeline,
    bank: MorganFeatureBank,
    task: Task,
    config: RFConfig,
) -> RFCVResult:
    """Fit one RF per fold and collect scaffold-held-out predictions.

    Folds are inherited from the frozen training-only pipeline. This function
    never tunes hyperparameters or fits a probability calibrator.
    """
    if task not in ("egfr", "her2", "dual"):
        raise ValueError(f"Unsupported task: {task}")
    if (
        bank.radius != config.radius
        or bank.n_bits != config.n_bits
        or bank.include_chirality != config.include_chirality
    ):
        raise ValueError("Fingerprint configuration differs from feature bank")

    training_ids = {record.structure_id for record in pipeline.data.train}
    if set(bank.molecule_ids) != training_ids:
        raise ValueError("Feature bank must contain exactly the training molecules")
    if len(pipeline.folds) != len(pipeline.fold_audits):
        raise ValueError("Fold audits do not match scaffold folds")

    fold_metrics: list[FoldMetrics] = []
    predictions: list[OOFPrediction] = []

    for fold, audit in zip(pipeline.folds, pipeline.fold_audits, strict=True):
        if fold.number != audit.number:
            raise ValueError("Fold and audit numbers are misaligned")
        ready = {
            "egfr": audit.egfr_regression_ready,
            "her2": audit.her2_regression_ready,
            "dual": audit.dual_classification_ready,
        }[task]
        if not ready:
            raise ValueError(f"Fold {fold.number} is not ready for {task}")

        fit = _examples(fold.fit, task, bank)
        holdout = _examples(fold.holdout, task, bank)

        if task == "dual":
            if not isinstance(fit, DualActivityExamples) or not isinstance(
                holdout, DualActivityExamples
            ):
                raise TypeError("Expected dual-activity examples")
            model = fit_dual_activity_classifier(fit, config)
            train_result, _ = evaluate_dual_activity_classifier(model, fit)
            holdout_result, scores = evaluate_dual_activity_classifier(model, holdout)
            observed = holdout.dual_active
            train_score = float(train_result["average_precision"])
            holdout_score = float(holdout_result["average_precision"])
            positives = int(holdout_result["positives"])
            prevalence = float(holdout_result["prevalence"])
            spearman = None
        else:
            if not isinstance(fit, TargetExamples) or not isinstance(
                holdout, TargetExamples
            ):
                raise TypeError("Expected target-specific regression examples")
            model = fit_random_forest(fit, config)
            train_result, _ = evaluate_random_forest(model, fit)
            holdout_result, scores = evaluate_random_forest(model, holdout)
            observed = holdout.pic50
            train_score = float(train_result["mae"])
            holdout_score = float(holdout_result["mae"])
            positives = None
            prevalence = None
            spearman_value = holdout_result["spearman"]
            spearman = float(spearman_value) if spearman_value is not None else None

        fold_metrics.append(
            FoldMetrics(
                fold=fold.number,
                n_fit=len(fit.molecule_ids),
                n_holdout=len(holdout.molecule_ids),
                train_score=train_score,
                holdout_score=holdout_score,
                holdout_spearman=spearman,
                holdout_positives=positives,
                holdout_prevalence=prevalence,
            )
        )
        predictions.extend(
            OOFPrediction(
                structure_id=structure_id,
                scaffold_group_id=scaffold_id,
                fold=fold.number,
                observed=float(observation),
                predicted=float(score),
            )
            for structure_id, scaffold_id, observation, score in zip(
                holdout.molecule_ids,
                holdout.scaffold_ids,
                observed,
                scores,
                strict=True,
            )
        )

    eligible = select_task_records(pipeline.data.train, task)
    by_id = {item.structure_id: item for item in predictions}
    if len(by_id) != len(predictions) or set(by_id) != {
        record.structure_id for record in eligible
    }:
        raise ValueError(
            "OOF predictions do not cover every eligible training molecule once"
        )

    # Preserve the frozen training row order for downstream calibration.
    ordered = tuple(by_id[record.structure_id] for record in eligible)
    actual = np.asarray([item.observed for item in ordered], dtype=float)
    estimated = np.asarray([item.predicted for item in ordered], dtype=float)
    if task == "dual":
        pooled_score = float(average_precision_score(actual.astype(int), estimated))
        metric_name: Literal["mae", "average_precision"] = "average_precision"
    else:
        pooled_score = float(mean_absolute_error(actual, estimated))
        metric_name = "mae"

    return RFCVResult(
        task=task,
        metric_name=metric_name,
        fold_metrics=tuple(fold_metrics),
        oof_predictions=ordered,
        mean_holdout_score=float(
            np.mean([item.holdout_score for item in fold_metrics])
        ),
        pooled_oof_score=pooled_score,
    )
