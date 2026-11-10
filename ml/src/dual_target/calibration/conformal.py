"""Stage C.5: scaffold-disjoint split-conformal pIC50 prediction intervals.

For each target, fit a fresh Random Forest on the frozen Stage C ``fit`` role,
then compute absolute prediction residuals on the disjoint ``calibration``
role. The conformal half-width is the finite-sample corrected empirical
quantile of those residuals. Do not use ``threshold`` records for fitting or
calibration; development validation is *evaluation only*; M2.1 test is locked.

This is a new Stage C subset-fit model, not the full-training Stage B forest.
Classical finite-sample marginal coverage requires exchangeable calibration
and evaluation observations. Chemically novel scaffold groups and selection of
hyperparameters using the original training partition can challenge those
assumptions. Report *empirical* held-out coverage, not guaranteed OOD coverage.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from ..data.modeling_dataset import MoleculeRecord
from ..data.pipeline import ModelingPipeline, select_task_records
from ..features.morgan_pipeline import MorganFeatureBank
from ..models.random_forest import RFConfig, TargetExamples, fit_random_forest
from .probability import _validate_frozen_split
from .scaffold_calibration import ScaffoldCalibrationSplit

PotencyTarget = Literal["egfr", "her2"]


@dataclass(frozen=True)
class ConformalMetrics:
    """Uncertainty diagnostics for a labeled, held-out subset."""

    n: int
    mae: float
    empirical_coverage: float
    mean_interval_width: float
    median_interval_width: float
    target_coverage: float


@dataclass(frozen=True)
class PotencyIntervalScores:
    """Aligned observed pIC50 values and conformal prediction intervals."""

    molecule_ids: tuple[str, ...]
    scaffold_ids: tuple[str, ...]
    observed_pic50: np.ndarray
    predicted_pic50: np.ndarray
    lower_pic50: np.ndarray
    upper_pic50: np.ndarray
    covered: np.ndarray
    metrics: ConformalMetrics


@dataclass(frozen=True)
class SplitConformalRegressor:
    """Subset-fit RF and single global residual-based interval half-width."""

    target: PotencyTarget
    forest: RandomForestRegressor
    config: RFConfig
    alpha: float
    half_width: float
    calibration_order_statistic: int
    n_fit_labels: int
    n_calibration_labels: int
    calibration_scores: PotencyIntervalScores
    assignment_sha256: str
    dataset_sha256: str
    manifest_sha256: str
    feature_sha256: str
    seed_used: int
    training_molecule_ids: frozenset[str]
    training_scaffold_ids: frozenset[str]

    def predict_intervals(
        self, features: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return point, lower, and upper pIC50 predictions, in row order."""
        matrix = _validated_features(features, self.config.n_bits)
        predicted = np.asarray(self.forest.predict(matrix), dtype=float)
        if predicted.shape != (len(matrix),) or not np.isfinite(predicted).all():
            raise ValueError("RF produced invalid pIC50 predictions")
        return predicted, predicted - self.half_width, predicted + self.half_width


def conformal_absolute_quantile(
    absolute_residuals: Sequence[float] | np.ndarray, *, alpha: float = 0.10
) -> tuple[float, int]:
    """Return split-conformal radius and 1-based corrected order statistic.

    For n calibration residuals, k = ceil((n + 1) * (1 - alpha)).
    The k-th smallest absolute residual is selected (not interpolation).
    If k > n, a *finite* distribution-free radius cannot be formed, and we
    reject the request rather than silently using an optimistic quantile.
    """
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
        raise TypeError("alpha must be a finite number in (0, 1)")
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be a finite number in (0, 1)")
    values = np.asarray(absolute_residuals, dtype=float)
    if (
        values.ndim != 1
        or len(values) == 0
        or not np.isfinite(values).all()
        or np.any(values < 0)
    ):
        raise ValueError("Residuals must be a nonempty finite nonnegative vector")
    k = math.ceil((len(values) + 1) * (1 - alpha))
    if k > len(values):
        raise ValueError("Too few calibration labels for a finite interval at alpha")
    radius = float(np.partition(values, k - 1)[k - 1])
    return radius, k


def _validated_features(features: np.ndarray, n_bits: int) -> np.ndarray:
    matrix = np.asarray(features)
    if matrix.ndim != 2 or len(matrix) == 0 or matrix.shape[1] != n_bits:
        raise ValueError("Expected a nonempty 2-D Morgan fingerprint matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("Morgan fingerprints contain nonfinite values")
    return matrix


def _target_examples(
    records: Sequence[MoleculeRecord],
    target: PotencyTarget,
    bank: MorganFeatureBank,
) -> TargetExamples:
    eligible = tuple(select_task_records(records, target))
    if not eligible:
        raise ValueError(f"No accepted {target.upper()} potency labels")
    potency = np.asarray(
        [getattr(record, f"{target}_pic50") for record in eligible],
        dtype=float,
    )
    if not np.isfinite(potency).all():
        raise ValueError(f"Nonfinite accepted {target.upper()} potency labels")
    return TargetExamples(
        molecule_ids=tuple(row.structure_id for row in eligible),
        scaffold_ids=tuple(row.scaffold_group_id for row in eligible),
        features=bank.select_features(eligible),
        pic50=potency,
    )


def _interval_scores(
    examples: TargetExamples,
    predicted: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    alpha: float,
) -> PotencyIntervalScores:
    observed = np.asarray(examples.pic50, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if not all(
        arr.ndim == 1 and len(arr) == len(observed) and np.isfinite(arr).all()
        for arr in (observed, predicted, lower, upper)
    ):
        raise ValueError("Observed values and interval boundaries must align")
    if np.any(lower > predicted) or np.any(predicted > upper):
        raise ValueError("Prediction lies outside its conformal interval")
    covered = (observed >= lower) & (observed <= upper)
    widths = upper - lower
    return PotencyIntervalScores(
        molecule_ids=examples.molecule_ids,
        scaffold_ids=examples.scaffold_ids,
        observed_pic50=observed,
        predicted_pic50=predicted,
        lower_pic50=lower,
        upper_pic50=upper,
        covered=covered,
        metrics=ConformalMetrics(
            n=len(observed),
            mae=float(np.mean(np.abs(observed - predicted))),
            empirical_coverage=float(np.mean(covered)),
            mean_interval_width=float(np.mean(widths)),
            median_interval_width=float(np.median(widths)),
            target_coverage=1.0 - alpha,
        ),
    )


def fit_split_conformal_regression(
    pipeline: ModelingPipeline,
    split: ScaffoldCalibrationSplit,
    train_bank: MorganFeatureBank,
    config: RFConfig,
    target: PotencyTarget,
    *,
    alpha: float = 0.10,
    expected_assignment_sha256: str,
) -> SplitConformalRegressor:
    """Fit on `fit`, calibrate pIC50 intervals on `calibration` only.

    `config` must be predetermined rather than chosen by calibration or
    threshold scores. Dataset, feature, manifest, and assignment checks reuse
    the Stage C.4 tested split-integrity guard.
    """
    if target not in ("egfr", "her2"):
        raise ValueError("target must be 'egfr' or 'her2'")
    _validate_frozen_split(
        pipeline, split, train_bank, config, expected_assignment_sha256
    )
    fit_examples = _target_examples(split.fit, target, train_bank)
    calibration_examples = _target_examples(split.calibration, target, train_bank)
    _validated_features(fit_examples.features, config.n_bits)
    _validated_features(calibration_examples.features, config.n_bits)

    model = fit_random_forest(fit_examples, config)
    calibration_pred = np.asarray(
        model.predict(calibration_examples.features), dtype=float
    )
    if (
        calibration_pred.shape != calibration_examples.pic50.shape
        or not np.isfinite(calibration_pred).all()
    ):
        raise ValueError("RF produced invalid calibration predictions")
    residuals = np.abs(calibration_examples.pic50 - calibration_pred)
    half_width, order_statistic = conformal_absolute_quantile(
        residuals, alpha=alpha
    )
    calibration_scores = _interval_scores(
        calibration_examples,
        calibration_pred,
        calibration_pred - half_width,
        calibration_pred + half_width,
        alpha=alpha,
    )
    return SplitConformalRegressor(
        target=target,
        forest=model,
        config=config,
        alpha=alpha,
        half_width=half_width,
        calibration_order_statistic=order_statistic,
        n_fit_labels=len(fit_examples.pic50),
        n_calibration_labels=len(calibration_examples.pic50),
        calibration_scores=calibration_scores,
        assignment_sha256=split.assignment_sha256,
        dataset_sha256=split.dataset_sha256,
        manifest_sha256=split.manifest_sha256,
        feature_sha256=train_bank.fingerprint_sha256,
        seed_used=split.seed_used,
        training_molecule_ids=frozenset(r.structure_id for r in pipeline.data.train),
        training_scaffold_ids=frozenset(
            r.scaffold_group_id for r in pipeline.data.train
        ),
    )


def evaluate_conformal_development_validation(
    fitted: SplitConformalRegressor,
    pipeline: ModelingPipeline,
    validation_bank: MorganFeatureBank,
) -> PotencyIntervalScores:
    """Score the frozen development-validation split without fitting/refitting."""
    if (
        fitted.dataset_sha256 != pipeline.data.dataset_sha256
        or fitted.manifest_sha256 != pipeline.data.manifest_sha256
    ):
        raise ValueError("Development-validation dataset provenance mismatch")
    records = tuple(pipeline.data.validation)
    if validation_bank.molecule_ids != tuple(r.structure_id for r in records):
        raise ValueError("Development-validation feature bank row mismatch")
    if (
        validation_bank.radius,
        validation_bank.n_bits,
        validation_bank.include_chirality,
    ) != (
        fitted.config.radius,
        fitted.config.n_bits,
        fitted.config.include_chirality,
    ):
        raise ValueError("Development-validation fingerprint configuration mismatch")
    if any(
        r.structure_id in fitted.training_molecule_ids
        or r.scaffold_group_id in fitted.training_scaffold_ids
        for r in records
    ):
        raise ValueError("Training molecule/scaffold leakage into development validation")
    examples = _target_examples(records, fitted.target, validation_bank)
    predicted, lower, upper = fitted.predict_intervals(examples.features)
    return _interval_scores(examples, predicted, lower, upper, alpha=fitted.alpha)
