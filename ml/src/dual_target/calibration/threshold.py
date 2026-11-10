"""Stage C.6: choose a dual-activity decision cutoff without label leakage.

Use scores on the frozen Stage C ``threshold`` role to select a threshold.
Apply that *same frozen* threshold to M2.1 development validation. The
``fit`` and ``calibration`` roles never participate in threshold selection;
M2.1 final-test rows are inaccessible to this module.

The decision score is the existing Stage C.4 forest's *raw* positive-class
probability by default. A sigmoid score is supported for protocol comparisons,
but callers must choose the score source before viewing threshold or
validation performance. Neither option refits the forest or sigmoid.

Ground truth remains unchanged: BOTH EGFR and HER2 pIC50 must be > 6.0.
The threshold is a cutoff on a model score, not an IC50 activity cutoff.

F1 is the predeclared optimization objective. When multiple cutoffs tie,
select the *highest* cutoff (a deterministic, conservative tie-break).
F1 depends on positive-class prevalence and may not reflect practical
compound-screening priorities; those must be decided independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..data.pipeline import ModelingPipeline
from ..features.morgan_pipeline import MorganFeatureBank
from .probability import (
    SigmoidDualCalibration,
    _make_paired_examples,
    _validate_frozen_split,
    evaluate_development_validation,
)
from .scaffold_calibration import ScaffoldCalibrationSplit

ScoreKind = Literal["raw", "calibrated"]
SubsetKind = Literal["threshold", "validation"]


@dataclass(frozen=True)
class ThresholdMetrics:
    """Binary classification diagnostics for a fixed cutoff (score >= cutoff)."""

    threshold: float
    n: int
    positives: int
    negatives: int
    predicted_positives: int
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    specificity: float
    f1: float
    balanced_accuracy: float
    accuracy: float


@dataclass(frozen=True)
class ThresholdPredictions:
    """Scores and decisions aligned to labeled molecules, ready for CSV export."""

    subset: SubsetKind
    molecule_ids: tuple[str, ...]
    scaffold_ids: tuple[str, ...]
    observed: np.ndarray
    scores: np.ndarray
    predicted: np.ndarray
    metrics: ThresholdMetrics


@dataclass(frozen=True)
class DualThresholdSelection:
    """Frozen threshold, its selection curve, and source-model provenance."""

    objective: Literal["f1"]
    score_kind: ScoreKind
    threshold: float
    selected_metrics: ThresholdMetrics
    curve: tuple[ThresholdMetrics, ...]
    threshold_predictions: ThresholdPredictions
    assignment_sha256: str
    dataset_sha256: str
    manifest_sha256: str
    train_feature_sha256: str
    seed_used: int
    n_fit_labels: int
    n_calibration_labels: int


def _validated_binary_labels(observed: np.ndarray) -> np.ndarray:
    y = np.asarray(observed)
    if (
        y.ndim != 1
        or len(y) < 2
        or y.dtype.kind not in "biuf"
        or not np.isfinite(y).all()
        or not np.all((y == 0) | (y == 1))
        or set(np.unique(y)) != {0, 1}
    ):
        raise ValueError("Threshold evaluation requires both binary classes (0, 1)")
    return y.astype(np.int8, copy=False)


def _validated_scores(scores: np.ndarray, n: int) -> np.ndarray:
    values = np.asarray(scores)
    if (
        values.ndim != 1
        or len(values) != n
        or values.dtype.kind not in "iuf"
        or not np.isfinite(values).all()
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError("Expected finite positive-class scores in [0, 1]")
    return values.astype(float, copy=False)


def _validated_threshold(threshold: float) -> float:
    if (
        isinstance(threshold, (bool, np.bool_))
        or not isinstance(threshold, (int, float, np.integer, np.floating))
        or not np.isfinite(threshold)
        or not 0.0 <= threshold <= 1.0
    ):
        raise ValueError("Decision threshold must be a finite number in [0, 1]")
    return float(threshold)


def _checked_arrays(
    observed: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    y = _validated_binary_labels(observed)
    return y, _validated_scores(scores, len(y))


def evaluate_fixed_threshold(
    observed: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> ThresholdMetrics:
    """Evaluate one threshold without fitting or selecting on these labels."""
    y, p = _checked_arrays(observed, scores)
    cutoff = _validated_threshold(threshold)
    predicted = p >= cutoff
    positive = y == 1
    tp = int(np.count_nonzero(predicted & positive))
    fp = int(np.count_nonzero(predicted & ~positive))
    fn = int(np.count_nonzero(~predicted & positive))
    tn = int(np.count_nonzero(~predicted & ~positive))
    n_pos = tp + fn
    n_neg = tn + fp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / n_pos
    specificity = tn / n_neg
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return ThresholdMetrics(
        threshold=cutoff,
        n=len(y),
        positives=n_pos,
        negatives=n_neg,
        predicted_positives=tp + fp,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        precision=float(precision),
        recall=float(recall),
        specificity=float(specificity),
        f1=float(f1),
        balanced_accuracy=float((recall + specificity) / 2.0),
        accuracy=float((tp + tn) / len(y)),
    )


def sweep_f1_thresholds(
    observed: np.ndarray,
    scores: np.ndarray,
) -> tuple[ThresholdMetrics, ...]:
    """Evaluate every distinct observed score as an inclusive cutoff.

    Since the rule is ``score >= threshold``, class predictions can only
    change at observed score values. The returned curve is sorted from
    *highest to lowest* cutoff. Exhaustive search avoids an arbitrary grid.
    """
    y, p = _checked_arrays(observed, scores)
    candidates = np.unique(p)[::-1]
    return tuple(evaluate_fixed_threshold(y, p, float(t)) for t in candidates)


def choose_f1_threshold(
    observed: np.ndarray,
    scores: np.ndarray,
) -> tuple[ThresholdMetrics, tuple[ThresholdMetrics, ...]]:
    """Maximize F1; on exact ties select the highest cutoff."""
    curve = sweep_f1_thresholds(observed, scores)
    chosen = max(curve, key=lambda row: (row.f1, row.threshold))
    return chosen, curve


def _source_scores(
    model: SigmoidDualCalibration, features: np.ndarray, kind: ScoreKind
) -> np.ndarray:
    if kind == "raw":
        return model.predict_raw_positive(features)
    if kind == "calibrated":
        return model.predict_calibrated_positive(features)
    raise ValueError("score_kind must be 'raw' or 'calibrated'")


def _prediction_rows(
    *,
    subset: SubsetKind,
    molecule_ids: tuple[str, ...],
    scaffold_ids: tuple[str, ...],
    observed: np.ndarray,
    scores: np.ndarray,
    metrics: ThresholdMetrics,
) -> ThresholdPredictions:
    y, p = _checked_arrays(observed, scores)
    if (
        len(molecule_ids) != len(y)
        or len(scaffold_ids) != len(y)
        or len(set(molecule_ids)) != len(y)
    ):
        raise ValueError("Threshold prediction identities do not align with labels")
    return ThresholdPredictions(
        subset=subset,
        molecule_ids=molecule_ids,
        scaffold_ids=scaffold_ids,
        observed=y.copy(),
        scores=p.copy(),
        predicted=(p >= metrics.threshold).astype(np.int8),
        metrics=metrics,
    )


def _check_source_model(
    model: SigmoidDualCalibration,
    pipeline: ModelingPipeline,
    split: ScaffoldCalibrationSplit,
    train_bank: MorganFeatureBank,
    expected_assignment_sha256: str,
) -> None:
    """Reuse Stage C integrity checks; bind selection to the fitted model."""
    _validate_frozen_split(
        pipeline, split, train_bank, model.config, expected_assignment_sha256
    )
    if (
        model.assignment_sha256 != split.assignment_sha256
        or model.dataset_sha256 != split.dataset_sha256
        or model.manifest_sha256 != split.manifest_sha256
        or model.feature_sha256 != train_bank.fingerprint_sha256
        or model.seed_used != split.seed_used
        or model.n_fit_labels != split.fit_audit.paired_labels
        or model.n_calibration_labels != split.calibration_audit.paired_labels
        or model.training_molecule_ids
        != frozenset(r.structure_id for r in pipeline.data.train)
        or model.training_scaffold_ids
        != frozenset(r.scaffold_group_id for r in pipeline.data.train)
    ):
        raise ValueError("Stage C threshold/model provenance mismatch")


def select_dual_threshold(
    model: SigmoidDualCalibration,
    pipeline: ModelingPipeline,
    split: ScaffoldCalibrationSplit,
    train_bank: MorganFeatureBank,
    *,
    expected_assignment_sha256: str,
    score_kind: ScoreKind = "raw",
) -> DualThresholdSelection:
    """Select a threshold ONLY on the reserved scaffold-disjoint role.

    This does not fit or update the classifier or sigmoid. The raw-score
    protocol was chosen after Stage C.4 validation and must be reported as
    a development-stage choice, not an unbiased performance comparison.
    """
    if score_kind not in ("raw", "calibrated"):
        raise ValueError("score_kind must be 'raw' or 'calibrated'")
    _check_source_model(
        model, pipeline, split, train_bank, expected_assignment_sha256
    )
    examples = _make_paired_examples(split.threshold, train_bank)
    scores = _source_scores(model, examples.features, score_kind)
    chosen, curve = choose_f1_threshold(examples.dual_active, scores)
    rows = _prediction_rows(
        subset="threshold",
        molecule_ids=examples.molecule_ids,
        scaffold_ids=examples.scaffold_ids,
        observed=examples.dual_active,
        scores=scores,
        metrics=chosen,
    )
    return DualThresholdSelection(
        objective="f1",
        score_kind=score_kind,
        threshold=chosen.threshold,
        selected_metrics=chosen,
        curve=curve,
        threshold_predictions=rows,
        assignment_sha256=split.assignment_sha256,
        dataset_sha256=split.dataset_sha256,
        manifest_sha256=split.manifest_sha256,
        train_feature_sha256=train_bank.fingerprint_sha256,
        seed_used=split.seed_used,
        n_fit_labels=model.n_fit_labels,
        n_calibration_labels=model.n_calibration_labels,
    )


def evaluate_frozen_dual_threshold(
    selection: DualThresholdSelection,
    model: SigmoidDualCalibration,
    pipeline: ModelingPipeline,
    validation_bank: MorganFeatureBank,
) -> ThresholdPredictions:
    """Apply the preselected cutoff on development validation; never retune."""
    if (
        selection.assignment_sha256 != model.assignment_sha256
        or selection.dataset_sha256 != model.dataset_sha256
        or selection.manifest_sha256 != model.manifest_sha256
        or selection.train_feature_sha256 != model.feature_sha256
        or selection.seed_used != model.seed_used
        or selection.n_fit_labels != model.n_fit_labels
        or selection.n_calibration_labels != model.n_calibration_labels
        or selection.objective != "f1"
        or selection.score_kind not in ("raw", "calibrated")
        or selection.threshold != selection.selected_metrics.threshold
    ):
        raise ValueError("Frozen threshold provenance does not match source model")
    _validated_threshold(selection.threshold)
    # Shared Stage C.4 evaluator independently verifies dataset identity,
    # molecule row order, feature configuration, and scaffold separation.
    validation_scores = evaluate_development_validation(
        model, pipeline, validation_bank
    )
    scores = (
        validation_scores.raw_probabilities
        if selection.score_kind == "raw"
        else validation_scores.calibrated_probabilities
    )
    metrics = evaluate_fixed_threshold(
        validation_scores.observed, scores, selection.threshold
    )
    return _prediction_rows(
        subset="validation",
        molecule_ids=validation_scores.molecule_ids,
        scaffold_ids=validation_scores.scaffold_ids,
        observed=validation_scores.observed,
        scores=scores,
        metrics=metrics,
    )
