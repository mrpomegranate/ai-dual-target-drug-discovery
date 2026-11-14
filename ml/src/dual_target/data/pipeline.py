
"""Composable, label-aware data preparation for M2 modeling.

Internal cross-validation uses the frozen training partition only.
Test labels are never loaded or exposed by this module.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..evaluation.scaffold_cv import ScaffoldFold, make_scaffold_folds
from .modeling_dataset import ModelingData, MoleculeRecord, load_modeling_data

Task = Literal["egfr", "her2", "dual"]
PIC50_ACTIVITY_THRESHOLD = 6.0


@dataclass(frozen=True)
class LabelCounts:
    """Accepted label coverage within a set of molecules."""

    molecules: int
    egfr_labels: int
    her2_labels: int
    paired_labels: int
    dual_active: int
    not_dual_active: int


@dataclass(frozen=True)
class FoldAudit:
    """Label coverage and modeling eligibility for one CV fold."""

    number: int
    fit: LabelCounts
    holdout: LabelCounts
    egfr_regression_ready: bool
    her2_regression_ready: bool
    dual_classification_ready: bool


@dataclass(frozen=True)
class ModelingPipeline:
    """Frozen data, internal scaffold folds, and label-coverage audits."""

    data: ModelingData
    folds: tuple[ScaffoldFold, ...]
    train_counts: LabelCounts
    validation_counts: LabelCounts
    fold_audits: tuple[FoldAudit, ...]


def _check_target(
    mask: int,
    label: float | None,
    target: str,
) -> None:
    """Validate an accepted or unavailable target measurement."""
    if mask not in (0, 1):
        raise ValueError(f"Invalid {target} mask: {mask!r}")

    if mask == 0 and label is not None:
        raise ValueError(f"Masked {target} target has a pIC50 label")

    if mask == 1 and (label is None or not math.isfinite(label)):
        raise ValueError(f"Accepted {target} label is missing or nonfinite")


def summarize_labels(records: Sequence[MoleculeRecord]) -> LabelCounts:
    """Count accepted labels without treating missing labels as inactive."""
    egfr_count = 0
    her2_count = 0
    paired_count = 0
    dual_active = 0

    for record in records:
        _check_target(record.egfr_mask, record.egfr_pic50, "EGFR")
        _check_target(record.her2_mask, record.her2_pic50, "HER2")

        if record.egfr_mask == 0 and record.her2_mask == 0:
            raise ValueError(
                f"Molecule has no accepted targets: {record.structure_id}"
            )

        egfr_count += record.egfr_mask
        her2_count += record.her2_mask

        if record.egfr_mask == 1 and record.her2_mask == 1:
            paired_count += 1

            if (
                record.egfr_pic50 > PIC50_ACTIVITY_THRESHOLD
                and record.her2_pic50 > PIC50_ACTIVITY_THRESHOLD
            ):
                dual_active += 1

    return LabelCounts(
        molecules=len(records),
        egfr_labels=egfr_count,
        her2_labels=her2_count,
        paired_labels=paired_count,
        dual_active=dual_active,
        not_dual_active=paired_count - dual_active,
    )


def select_task_records(
    records: Sequence[MoleculeRecord],
    task: Task,
) -> tuple[MoleculeRecord, ...]:
    """Select accepted records for regression or paired classification."""
    if task == "egfr":
        return tuple(
            record for record in records if record.egfr_mask == 1
        )

    if task == "her2":
        return tuple(
            record for record in records if record.her2_mask == 1
        )

    if task == "dual":
        return tuple(
            record
            for record in records
            if record.egfr_mask == 1 and record.her2_mask == 1
        )

    raise ValueError(f"Unknown modeling task: {task}")


def _has_both_classes(counts: LabelCounts) -> bool:
    return counts.dual_active > 0 and counts.not_dual_active > 0


def compose_modeling_pipeline(
    data: ModelingData,
    n_splits: int = 5,
) -> ModelingPipeline:
    """Compose frozen M2.1 data with scaffold CV and label audits."""
    train_counts = summarize_labels(data.train)
    validation_counts = summarize_labels(data.validation)

    folds = make_scaffold_folds(data, n_splits=n_splits)
    audits = []

    for fold in folds:
        fit = summarize_labels(fold.fit)
        holdout = summarize_labels(fold.holdout)

        audits.append(
            FoldAudit(
                number=fold.number,
                fit=fit,
                holdout=holdout,
                egfr_regression_ready=(
                    fit.egfr_labels > 0 and holdout.egfr_labels > 0
                ),
                her2_regression_ready=(
                    fit.her2_labels > 0 and holdout.her2_labels > 0
                ),
                dual_classification_ready=(
                    _has_both_classes(fit)
                    and _has_both_classes(holdout)
                ),
            )
        )

    if sum(a.holdout.molecules for a in audits) != train_counts.molecules:
        raise ValueError("Internal CV holdouts do not cover the training set")

    if sum(a.holdout.egfr_labels for a in audits) != train_counts.egfr_labels:
        raise ValueError("EGFR labels are missing from internal CV holdouts")

    if sum(a.holdout.her2_labels for a in audits) != train_counts.her2_labels:
        raise ValueError("HER2 labels are missing from internal CV holdouts")

    if sum(a.holdout.paired_labels for a in audits) != train_counts.paired_labels:
        raise ValueError("Paired labels are missing from internal CV holdouts")

    return ModelingPipeline(
        data=data,
        folds=folds,
        train_counts=train_counts,
        validation_counts=validation_counts,
        fold_audits=tuple(audits),
    )


def load_modeling_pipeline(
    dataset_path: str | Path,
    manifest_path: str | Path,
    summary_path: str | Path,
    n_splits: int = 5,
) -> ModelingPipeline:
    """Load frozen data and construct an auditable modeling pipeline."""
    data = load_modeling_data(
        dataset_path,
        manifest_path,
        summary_path,
    )
    return compose_modeling_pipeline(data, n_splits=n_splits)

# --- Fit-only model-selection boundary (Stage C, audited seed 719) ---
# Keep this boundary in data/pipeline.py. Optuna callers must use ONLY the
# returned .data.train and build a NEW feature bank for those records.

FROZEN_STAGE_C_ASSIGNMENT_SHA256 = (
    "2b99214bce37d30e1eed8f6e22984242492560b2999bb174a8d378604de9171c"
)


@dataclass(frozen=True)
class FitOnlyData:
    """Only the model-fitting records; no calibration/threshold labels exposed."""

    train: tuple
    dataset_sha256: str
    manifest_sha256: str


@dataclass(frozen=True)
class FitOnlyFoldAudit:
    number: int
    egfr_regression_ready: bool
    her2_regression_ready: bool
    dual_classification_ready: bool


@dataclass(frozen=True)
class FitOnlyTuningPipeline:
    """Interface consumed by existing RF CV and Optuna code."""

    data: FitOnlyData
    folds: tuple
    fold_audits: tuple[FitOnlyFoldAudit, ...]
    stage_c_assignment_sha256: str
    stage_c_seed: int


def _fit_only_assignment_digest(split) -> str:
    import hashlib
    import json

    canonical = [
        [a.structure_id, a.scaffold_group_id, a.role]
        for a in sorted(split.assignments, key=lambda a: a.structure_id)
    ]
    return hashlib.sha256(
        json.dumps(
            canonical, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()


def _fit_only_check_integrity(source, split, expected_assignment_sha256):
    if split.assignment_sha256 != expected_assignment_sha256:
        raise ValueError("Stage C assignment differs from frozen seed-719 hash")
    if _fit_only_assignment_digest(split) != split.assignment_sha256:
        raise ValueError("Stage C assignment manifest content/hash mismatch")
    if (
        split.dataset_sha256 != source.data.dataset_sha256
        or split.manifest_sha256 != source.data.manifest_sha256
    ):
        raise ValueError("Stage C dataset/outer-manifest provenance mismatch")

    original = tuple(source.data.train)
    by_id = {record.structure_id: record for record in original}
    if len(by_id) != len(original):
        raise ValueError("Original training contains duplicate molecule IDs")
    recorded_roles = {}
    seen_ids = set()
    seen_scaffolds = set()
    for role, subset in (
        ("fit", split.fit),
        ("calibration", split.calibration),
        ("threshold", split.threshold),
    ):
        if not subset:
            raise ValueError(f"Empty Stage C {role} partition")
        ids = {row.structure_id for row in subset}
        scaffolds = {row.scaffold_group_id for row in subset}
        if len(ids) != len(subset) or seen_ids.intersection(ids):
            raise ValueError("Duplicate or overlapping Stage C molecule identities")
        if seen_scaffolds.intersection(scaffolds):
            raise ValueError("Stage C scaffold overlap")
        for row in subset:
            frozen = by_id.get(row.structure_id)
            if frozen is None or frozen != row:
                raise ValueError("Stage C record differs from frozen training data")
            recorded_roles[row.structure_id] = role
        seen_ids.update(ids)
        seen_scaffolds.update(scaffolds)

    if seen_ids != set(by_id) or len(split.assignments) != len(original):
        raise ValueError("Stage C partitions must cover original training exactly")
    for row in split.assignments:
        frozen = by_id.get(row.structure_id)
        if (
            frozen is None
            or row.scaffold_group_id != frozen.scaffold_group_id
            or row.role != recorded_roles[row.structure_id]
        ):
            raise ValueError("Stage C role manifest disagrees with assigned records")

    intended_fit = tuple(
        record for record in original
        if recorded_roles[record.structure_id] == "fit"
    )
    if tuple(split.fit) != intended_fit:
        raise ValueError("Stage C fitting records differ from frozen training order")

    validation = tuple(source.data.validation)
    val_ids = {record.structure_id for record in validation}
    val_scaffolds = {record.scaffold_group_id for record in validation}
    if seen_ids.intersection(val_ids) or seen_scaffolds.intersection(val_scaffolds):
        raise ValueError("Stage C training overlaps development validation")


def _fit_only_fold_ready(records, task):
    import numpy as np

    eligible = select_task_records(records, task)
    if len(eligible) < 2:
        return False
    if task in ("egfr", "her2"):
        return all(
            np.isfinite(getattr(row, f"{task}_pic50"))
            for row in eligible
        )
    if any(
        not (np.isfinite(row.egfr_pic50) and np.isfinite(row.her2_pic50))
        for row in eligible
    ):
        return False
    positive = sum(
        row.egfr_pic50 > PIC50_ACTIVITY_THRESHOLD
        and row.her2_pic50 > PIC50_ACTIVITY_THRESHOLD
        for row in eligible
    )
    return 0 < positive < len(eligible)


def build_fit_only_tuning_pipeline(
    source,
    split,
    *,
    n_splits=5,
    expected_assignment_sha256=FROZEN_STAGE_C_ASSIGNMENT_SHA256,
):
    """Make a scaffold-CV view restricted to Stage C fitting molecules only.

    Parameters
    ----------
    source : ModelingPipeline
        Original frozen M2.1 modeling pipeline. Never use its full training
        records for Optuna once the Stage C partition has been defined.
    split : ScaffoldCalibrationSplit
        Previously audited Stage C split, with assignment hash checked here.

    Returns
    -------
    FitOnlyTuningPipeline
        Has `.data.train`, `.folds`, and `.fold_audits` consumed by current
        `run_rf_scaffold_cv` and `optimize_rf`. It deliberately does NOT carry
        calibration, threshold, development validation, or test records.
    """
    import numpy as np
    from sklearn.model_selection import GroupKFold

    from ..evaluation.scaffold_cv import ScaffoldFold

    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("n_splits must be an integer >= 2")
    if not expected_assignment_sha256:
        raise ValueError("A frozen Stage C assignment hash is required")
    _fit_only_check_integrity(source, split, expected_assignment_sha256)

    fitting = tuple(split.fit)
    groups = np.asarray([record.scaffold_group_id for record in fitting])
    if len(set(groups)) < n_splits:
        raise ValueError("Not enough fitting scaffolds for the requested CV")

    folds = []
    audits = []
    observed_heldout = set()
    cv = GroupKFold(n_splits=n_splits)
    for number, (train_ix, holdout_ix) in enumerate(
        cv.split(np.arange(len(fitting)), groups=groups)
    ):
        fit_fold = tuple(fitting[int(i)] for i in train_ix)
        holdout = tuple(fitting[int(i)] for i in holdout_ix)
        if {r.scaffold_group_id for r in fit_fold}.intersection(
            r.scaffold_group_id for r in holdout
        ):
            raise ValueError("Internal CV scaffold leakage")
        heldout_ids = {r.structure_id for r in holdout}
        if observed_heldout.intersection(heldout_ids):
            raise ValueError("Repeated holdout molecule across CV folds")
        observed_heldout.update(heldout_ids)
        e = _fit_only_fold_ready(fit_fold, "egfr") and _fit_only_fold_ready(
            holdout, "egfr"
        )
        h = _fit_only_fold_ready(fit_fold, "her2") and _fit_only_fold_ready(
            holdout, "her2"
        )
        d = _fit_only_fold_ready(fit_fold, "dual") and _fit_only_fold_ready(
            holdout, "dual"
        )
        if not (e and h and d):
            raise ValueError(
                f"Fit-only fold {number} lacks required labels or both dual classes"
            )
        folds.append(ScaffoldFold(number=number, fit=fit_fold, holdout=holdout))
        audits.append(FitOnlyFoldAudit(number, e, h, d))

    if observed_heldout != {record.structure_id for record in fitting}:
        raise ValueError("Internal CV did not hold out each fitting molecule once")

    return FitOnlyTuningPipeline(
        data=FitOnlyData(
            train=fitting,
            dataset_sha256=source.data.dataset_sha256,
            manifest_sha256=source.data.manifest_sha256,
        ),
        folds=tuple(folds),
        fold_audits=tuple(audits),
        stage_c_assignment_sha256=split.assignment_sha256,
        stage_c_seed=split.seed_used,
    )
# --- End fit-only model-selection boundary ---
