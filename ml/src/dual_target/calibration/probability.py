"""Stage C.4: fit a sigmoid calibrator without reusing classifier-fit molecules.

Use only the frozen Stage C scaffold-disjoint ``fit`` and ``calibration``
subsets. The direct dual-activity Random Forest is fitted on ``fit``;
a one-dimensional Platt-style logistic sigmoid is fitted on its predictions
for ``calibration``. The ``threshold`` subset is not accessed by this module.

The ``calibration`` metrics returned here are *in-sample with respect to the
sigmoid*, not an unbiased measure of calibration quality. Use
``evaluate_development_validation`` to evaluate the frozen calibrated model
on M2.1 development validation. The M2.1 final test split is never accessed.

The Stage C model is a *new fit-subset forest*, not the full-training Stage B
forest. Original training records were involved in earlier Optuna selection;
scaffold separation here does not erase that model-selection dependency.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

from ..data.modeling_dataset import MoleculeRecord
from ..data.pipeline import (
    PIC50_ACTIVITY_THRESHOLD,
    ModelingPipeline,
    select_task_records,
)
from ..features.morgan_pipeline import MorganFeatureBank
from ..models.random_forest import (
    DualActivityExamples,
    RFConfig,
    fit_dual_activity_classifier,
)
from .scaffold_calibration import ScaffoldCalibrationSplit


@dataclass(frozen=True)
class ProbabilityMetrics:
    """Diagnostics on a labeled subset; lower Brier/log loss is better."""

    n: int
    positives: int
    positive_prevalence: float
    average_precision: float
    brier_score: float
    log_loss: float


@dataclass(frozen=True)
class DualProbabilityScores:
    """Row-aligned raw and sigmoid-calibrated positive-class probabilities."""

    molecule_ids: tuple[str, ...]
    scaffold_ids: tuple[str, ...]
    observed: np.ndarray
    raw_probabilities: np.ndarray
    calibrated_probabilities: np.ndarray
    raw_metrics: ProbabilityMetrics
    calibrated_metrics: ProbabilityMetrics


@dataclass(frozen=True)
class SigmoidDualCalibration:
    """Fresh RF + Platt-style sigmoid, with frozen Stage C provenance."""

    forest: RandomForestClassifier
    sigmoid: LogisticRegression
    config: RFConfig
    assignment_sha256: str
    dataset_sha256: str
    manifest_sha256: str
    feature_sha256: str
    seed_used: int
    n_fit_labels: int
    n_calibration_labels: int
    calibration_scores: DualProbabilityScores
    training_molecule_ids: frozenset[str]
    training_scaffold_ids: frozenset[str]

    def predict_raw_positive(self, features: np.ndarray) -> np.ndarray:
        """Predict the probability of *both* EGFR and HER2 being active."""
        matrix = _validated_features(features, self.config.n_bits)
        classes = list(self.forest.classes_)
        if classes != [0, 1]:
            raise ValueError("Expected Random Forest binary classes [0, 1]")
        raw = np.asarray(self.forest.predict_proba(matrix)[:, 1], dtype=float)
        _validated_probabilities(raw, len(matrix))
        return raw

    def predict_calibrated_positive(self, features: np.ndarray) -> np.ndarray:
        """Apply the fitted sigmoid to raw RF probabilities (not labels)."""
        raw = self.predict_raw_positive(features)
        probs = np.asarray(
            self.sigmoid.predict_proba(raw.reshape(-1, 1))[:, 1],
            dtype=float,
        )
        _validated_probabilities(probs, len(raw))
        return probs

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return [P(not dual-active), P(dual-active)] for each molecule."""
        positive = self.predict_calibrated_positive(features)
        return np.column_stack((1.0 - positive, positive))


def _validated_features(features: np.ndarray, n_bits: int) -> np.ndarray:
    matrix = np.asarray(features)
    if matrix.ndim != 2 or matrix.shape[1] != n_bits or len(matrix) == 0:
        raise ValueError("Expected a nonempty 2-D Morgan fingerprint matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("Molecular fingerprints contain nonfinite values")
    return matrix


def _validated_probabilities(probs: np.ndarray, n: int) -> None:
    if (
        probs.ndim != 1
        or len(probs) != n
        or not np.isfinite(probs).all()
        or np.any(probs < 0.0)
        or np.any(probs > 1.0)
    ):
        raise ValueError("Invalid positive-class probabilities")


def _make_paired_examples(
    records: Sequence[MoleculeRecord], bank: MorganFeatureBank
) -> DualActivityExamples:
    eligible = tuple(select_task_records(records, "dual"))
    if not eligible:
        raise ValueError("No accepted paired labels available")
    egfr = np.asarray([r.egfr_pic50 for r in eligible], dtype=float)
    her2 = np.asarray([r.her2_pic50 for r in eligible], dtype=float)
    if not np.isfinite(egfr).all() or not np.isfinite(her2).all():
        raise ValueError("Paired accepted pIC50 values must be finite")
    return DualActivityExamples(
        molecule_ids=tuple(r.structure_id for r in eligible),
        scaffold_ids=tuple(r.scaffold_group_id for r in eligible),
        features=bank.select_features(eligible),
        dual_active=np.asarray(
            (egfr > PIC50_ACTIVITY_THRESHOLD)
            & (her2 > PIC50_ACTIVITY_THRESHOLD),
            dtype=np.int8,
        ),
    )


def _scores_from_probabilities(
    examples: DualActivityExamples,
    raw: np.ndarray,
    calibrated: np.ndarray,
) -> DualProbabilityScores:
    y = np.asarray(examples.dual_active, dtype=np.int8)
    _validated_probabilities(raw, len(y))
    _validated_probabilities(calibrated, len(y))
    return DualProbabilityScores(
        molecule_ids=examples.molecule_ids,
        scaffold_ids=examples.scaffold_ids,
        observed=y,
        raw_probabilities=raw,
        calibrated_probabilities=calibrated,
        raw_metrics=_metrics(y, raw),
        calibrated_metrics=_metrics(y, calibrated),
    )


def _scores(
    model: SigmoidDualCalibration,
    examples: DualActivityExamples,
) -> DualProbabilityScores:
    raw = model.predict_raw_positive(examples.features)
    calibrated = model.predict_calibrated_positive(examples.features)
    return _scores_from_probabilities(examples, raw, calibrated)


def _metrics(observed: np.ndarray, probs: np.ndarray) -> ProbabilityMetrics:
    y = np.asarray(observed)
    _validated_probabilities(np.asarray(probs), len(y))
    if y.ndim != 1 or len(y) == 0 or set(np.unique(y)) != {0, 1}:
        raise ValueError("Probability evaluation requires both binary classes")
    return ProbabilityMetrics(
        n=len(y),
        positives=int(np.sum(y)),
        positive_prevalence=float(np.mean(y)),
        average_precision=float(average_precision_score(y, probs)),
        brier_score=float(brier_score_loss(y, probs)),
        log_loss=float(log_loss(y, probs, labels=[0, 1])),
    )


def _validate_frozen_split(
    pipeline: ModelingPipeline,
    split: ScaffoldCalibrationSplit,
    bank: MorganFeatureBank,
    config: RFConfig,
    expected_assignment_sha256: str,
) -> None:
    train = tuple(pipeline.data.train)
    validation = tuple(pipeline.data.validation)
    if (
        split.dataset_sha256 != pipeline.data.dataset_sha256
        or split.manifest_sha256 != pipeline.data.manifest_sha256
    ):
        raise ValueError("Stage C split dataset/manifest provenance mismatch")
    if split.assignment_sha256 != expected_assignment_sha256:
        raise ValueError("Stage C assignment hash does not match frozen audit")
    if not expected_assignment_sha256 or len(expected_assignment_sha256) != 64:
        raise ValueError("Expected an exact 64-character SHA-256 assignment hash")
    if bank.molecule_ids != tuple(r.structure_id for r in train):
        raise ValueError("Training feature bank must match frozen train row order")
    if (config.radius, config.n_bits, config.include_chirality) != (
        bank.radius, bank.n_bits, bank.include_chirality
    ):
        raise ValueError("RF and fingerprint-bank configurations differ")

    roles = {
        "fit": split.fit,
        "calibration": split.calibration,
        "threshold": split.threshold,
    }
    seen_ids: set[str] = set()
    seen_groups: set[str] = set()
    role_by_id: dict[str, str] = {}
    for role, records in roles.items():
        if not records:
            raise ValueError(f"Stage C {role} partition is empty")
        ids = {r.structure_id for r in records}
        groups = {r.scaffold_group_id for r in records}
        if len(ids) != len(records) or ids & seen_ids or groups & seen_groups:
            raise ValueError("Stage C partitions contain molecule or scaffold overlap")
        seen_ids.update(ids)
        seen_groups.update(groups)
        role_by_id.update({r.structure_id: role for r in records})
    train_by_id = {r.structure_id: r for r in train}
    if len(train_by_id) != len(train) or seen_ids != set(train_by_id):
        raise ValueError("Stage C split must cover only frozen train molecules")
    for records in roles.values():
        for row in records:
            original = train_by_id[row.structure_id]
            if row != original:
                raise ValueError("Stage C molecule records differ from frozen training")
    if any(
        r.structure_id in seen_ids or r.scaffold_group_id in seen_groups
        for r in validation
    ):
        raise ValueError("Development-validation molecule/scaffold leaked into Stage C")

    assignments = tuple(split.assignments)
    if (
        len(assignments) != len(train)
        or len({a.structure_id for a in assignments}) != len(train)
    ):
        raise ValueError("Incomplete or duplicated Stage C manifest assignments")
    if any(
        a.structure_id not in train_by_id
        or a.scaffold_group_id != train_by_id[a.structure_id].scaffold_group_id
        or a.role != role_by_id[a.structure_id]
        for a in assignments
    ):
        raise ValueError("Stage C assignment manifest does not match subsets")
    canonical = [
        [row.structure_id, row.scaffold_group_id, row.role]
        for row in sorted(assignments, key=lambda row: row.structure_id)
    ]
    digest = hashlib.sha256(
        json.dumps(canonical, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    if digest != split.assignment_sha256:
        raise ValueError("Stage C assignment hash does not match manifest contents")


def fit_sigmoid_dual_calibration(
    pipeline: ModelingPipeline,
    split: ScaffoldCalibrationSplit,
    train_bank: MorganFeatureBank,
    config: RFConfig,
    *,
    expected_assignment_sha256: str,
) -> SigmoidDualCalibration:
    """Fit fresh RF on 'fit', then sigmoid on 'calibration', never 'threshold'.

    Use the *preselected* classifier RF configuration. ``config`` must not
    be selected or altered using the calibration/threshold/validation metrics.
    Stage C contains a different, smaller-training forest than Stage B.
    """
    _validate_frozen_split(
        pipeline, split, train_bank, config, expected_assignment_sha256
    )
    fit_examples = _make_paired_examples(split.fit, train_bank)
    calibration_examples = _make_paired_examples(split.calibration, train_bank)
    if set(np.unique(fit_examples.dual_active)) != {0, 1}:
        raise ValueError("Fit subset requires both dual-activity classes")
    if set(np.unique(calibration_examples.dual_active)) != {0, 1}:
        raise ValueError("Calibration subset requires both dual-activity classes")

    forest = fit_dual_activity_classifier(fit_examples, config)
    if list(forest.classes_) != [0, 1]:
        raise ValueError("Expected fitted forest classes [0, 1]")
    raw = np.asarray(
        forest.predict_proba(calibration_examples.features)[:, 1],
        dtype=float,
    )
    _validated_probabilities(raw, len(calibration_examples.dual_active))
    sigmoid = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    sigmoid.fit(raw.reshape(-1, 1), calibration_examples.dual_active)
    if list(sigmoid.classes_) != [0, 1]:
        raise ValueError("Expected fitted sigmoid classes [0, 1]")

    calibrated = np.asarray(
        sigmoid.predict_proba(raw.reshape(-1, 1))[:, 1], dtype=float
    )
    calibration_scores = _scores_from_probabilities(
        calibration_examples, raw, calibrated
    )
    result = SigmoidDualCalibration(
        forest=forest,
        sigmoid=sigmoid,
        config=config,
        assignment_sha256=split.assignment_sha256,
        dataset_sha256=split.dataset_sha256,
        manifest_sha256=split.manifest_sha256,
        feature_sha256=train_bank.fingerprint_sha256,
        seed_used=split.seed_used,
        n_fit_labels=len(fit_examples.dual_active),
        n_calibration_labels=len(calibration_examples.dual_active),
        calibration_scores=calibration_scores,
        training_molecule_ids=frozenset(r.structure_id for r in pipeline.data.train),
        training_scaffold_ids=frozenset(
            r.scaffold_group_id for r in pipeline.data.train
        ),
    )
    return result


def evaluate_development_validation(
    calibrated: SigmoidDualCalibration,
    pipeline: ModelingPipeline,
    validation_bank: MorganFeatureBank,
) -> DualProbabilityScores:
    """Report untouched development-validation diagnostics; never refit."""
    if (
        calibrated.dataset_sha256 != pipeline.data.dataset_sha256
        or calibrated.manifest_sha256 != pipeline.data.manifest_sha256
    ):
        raise ValueError("Validation dataset provenance mismatch")
    records = tuple(pipeline.data.validation)
    if validation_bank.molecule_ids != tuple(r.structure_id for r in records):
        raise ValueError("Development-validation feature bank row mismatch")
    if (
        validation_bank.radius,
        validation_bank.n_bits,
        validation_bank.include_chirality,
    ) != (
        calibrated.config.radius,
        calibrated.config.n_bits,
        calibrated.config.include_chirality,
    ):
        raise ValueError("Validation fingerprint configuration mismatch")
    if any(
        r.structure_id in calibrated.training_molecule_ids
        or r.scaffold_group_id in calibrated.training_scaffold_ids
        for r in records
    ):
        raise ValueError("Train/development-validation molecule or scaffold overlap")
    examples = _make_paired_examples(records, validation_bank)
    return _scores(calibrated, examples)
