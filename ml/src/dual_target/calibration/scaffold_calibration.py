"""Make auditable, scaffold-disjoint calibration and decision subsets.

Stage C uses only the frozen M2.1 *training* partition. The three roles are:

* fit:        fit a fresh model using selected hyperparameters;
* calibration: fit a probability calibrator or interval correction;
* threshold: select a downstream decision threshold without reusing the
             molecules that fitted the calibrator.

This module only creates assignments and audits. It does not fit a model,
calibrate predictions, tune a threshold, or access the frozen test split.

Important: although these subsets are disjoint for Stage C fitting, all
original training molecules participated in the earlier Optuna CV search.
Consequently they are not independent of hyperparameter selection. Report
Stage C development-validation performance separately; do not claim a
pristine post-selection calibration sample or guaranteed conformal coverage.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from ..data.modeling_dataset import MoleculeRecord
from ..data.pipeline import (
    PIC50_ACTIVITY_THRESHOLD,
    ModelingPipeline,
    select_task_records,
)

CalibrationRole = Literal["fit", "calibration", "threshold"]


@dataclass(frozen=True)
class CalibrationAssignment:
    """One molecule's role in the training-only Stage C protocol."""

    structure_id: str
    scaffold_group_id: str
    role: CalibrationRole


@dataclass(frozen=True)
class CalibrationAudit:
    """Available accepted labels and paired-class counts in one subset."""

    molecules: int
    scaffolds: int
    egfr_labels: int
    her2_labels: int
    paired_labels: int
    dual_active: int
    not_dual_active: int


@dataclass(frozen=True)
class ScaffoldCalibrationSplit:
    """Disjoint Stage C subsets, audits, and reproducibility evidence."""

    fit: tuple[MoleculeRecord, ...]
    calibration: tuple[MoleculeRecord, ...]
    threshold: tuple[MoleculeRecord, ...]
    fit_audit: CalibrationAudit
    calibration_audit: CalibrationAudit
    threshold_audit: CalibrationAudit
    assignments: tuple[CalibrationAssignment, ...]
    assignment_sha256: str
    dataset_sha256: str
    manifest_sha256: str
    seed_used: int
    requested_calibration_fraction: float
    requested_threshold_fraction: float

    def manifest_rows(self) -> tuple[dict[str, str], ...]:
        """Ready for reproducible CSV/MLflow export; never contains labels."""
        return tuple(
            {
                "structure_id": row.structure_id,
                "scaffold_group_id": row.scaffold_group_id,
                "role": row.role,
            }
            for row in self.assignments
        )


def _validate_partition_integrity(
    train: tuple[MoleculeRecord, ...],
    validation: tuple[MoleculeRecord, ...],
) -> None:
    if not train:
        raise ValueError("Frozen training partition is empty")

    ids: set[str] = set()
    groups: set[str] = set()
    for record in train:
        molecule_id = record.structure_id
        scaffold = record.scaffold_group_id
        if not isinstance(molecule_id, str) or not molecule_id.strip():
            raise ValueError("Training structure_id must be a nonempty string")
        if not isinstance(scaffold, str) or not scaffold.strip():
            raise ValueError("Training scaffold_group_id must be a nonempty string")
        if molecule_id in ids:
            raise ValueError("Duplicate training molecule identity")
        ids.add(molecule_id)
        groups.add(scaffold)

    for record in validation:
        if record.structure_id in ids:
            raise ValueError("Training/development-validation molecule overlap")
        if record.scaffold_group_id in groups:
            raise ValueError("Training/development-validation scaffold overlap")

    if len(groups) < 3:
        raise ValueError("At least three distinct training scaffolds are required")

    # Only accepted exact-potency labels are used; a missing target is not
    # a negative dual-activity label. Censored/indeterminate labels were
    # excluded earlier in M1.
    for task in ("egfr", "her2"):
        for record in select_task_records(train, task):
            value = getattr(record, f"{task}_pic50")
            if not np.isfinite(value):
                raise ValueError(f"Nonfinite accepted {task} potency label")


def audit_calibration_records(
    records: tuple[MoleculeRecord, ...],
) -> CalibrationAudit:
    """Count available labels; apply the frozen strict pIC50 > 6 rule."""
    egfr = select_task_records(records, "egfr")
    her2 = select_task_records(records, "her2")
    paired = select_task_records(records, "dual")
    positive = sum(
        (record.egfr_pic50 > PIC50_ACTIVITY_THRESHOLD)
        and (record.her2_pic50 > PIC50_ACTIVITY_THRESHOLD)
        for record in paired
    )
    return CalibrationAudit(
        molecules=len(records),
        scaffolds=len({r.scaffold_group_id for r in records}),
        egfr_labels=len(egfr),
        her2_labels=len(her2),
        paired_labels=len(paired),
        dual_active=positive,
        not_dual_active=len(paired) - positive,
    )


def _ready(
    audit: CalibrationAudit,
    *,
    minimum_target_labels: int,
    minimum_dual_per_class: int,
) -> bool:
    return (
        audit.egfr_labels >= minimum_target_labels
        and audit.her2_labels >= minimum_target_labels
        and audit.dual_active >= minimum_dual_per_class
        and audit.not_dual_active >= minimum_dual_per_class
    )


def _assert_disjoint_complete(
    all_records: tuple[MoleculeRecord, ...],
    fit: tuple[MoleculeRecord, ...],
    calibration: tuple[MoleculeRecord, ...],
    threshold: tuple[MoleculeRecord, ...],
) -> None:
    original_ids = {record.structure_id for record in all_records}
    seen_ids: set[str] = set()
    seen_groups: set[str] = set()
    for name, records in (
        ("fit", fit),
        ("calibration", calibration),
        ("threshold", threshold),
    ):
        ids = {record.structure_id for record in records}
        groups = {record.scaffold_group_id for record in records}
        if not records:
            raise ValueError(f"Empty {name} partition")
        if ids & seen_ids:
            raise ValueError("Stage C molecule overlap across partitions")
        if groups & seen_groups:
            raise ValueError("Stage C scaffold overlap across partitions")
        seen_ids.update(ids)
        seen_groups.update(groups)
    if seen_ids != original_ids:
        raise ValueError(
            "Stage C assignments do not cover the frozen training partition"
        )


def _assignments_and_hash(
    all_records: tuple[MoleculeRecord, ...],
    fit: tuple[MoleculeRecord, ...],
    calibration: tuple[MoleculeRecord, ...],
    threshold: tuple[MoleculeRecord, ...],
) -> tuple[tuple[CalibrationAssignment, ...], str]:
    roles: dict[str, CalibrationRole] = {}
    for role, subset in (
        ("fit", fit),
        ("calibration", calibration),
        ("threshold", threshold),
    ):
        for record in subset:
            roles[record.structure_id] = role

    assignments = tuple(
        CalibrationAssignment(
            structure_id=record.structure_id,
            scaffold_group_id=record.scaffold_group_id,
            role=roles[record.structure_id],
        )
        for record in all_records
    )
    canonical = [
        [row.structure_id, row.scaffold_group_id, row.role]
        for row in sorted(assignments, key=lambda row: row.structure_id)
    ]
    digest = hashlib.sha256(
        json.dumps(canonical, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    return assignments, digest


def make_scaffold_calibration_split(
    pipeline: ModelingPipeline,
    *,
    calibration_fraction: float = 0.20,
    threshold_fraction: float = 0.20,
    seed: int = 714,
    minimum_target_labels: int = 10,
    minimum_dual_per_class: int = 5,
    max_attempts: int = 50,
) -> ScaffoldCalibrationSplit:
    """Split *only training molecules* into model-fit/calibration/threshold roles.

    GroupShuffleSplit samples scaffold groups rather than individual molecules.
    Thus the requested fractions are approximate **scaffold fractions**, not
    guaranteed exact molecule fractions. We retry deterministically only when
    a role lacks enough accepted target labels or dual-class examples.

    No score-based search is used to choose a split. Changing seed or any
    fraction creates a new calibration assignment and should be versioned.
    """
    if not (
        np.isfinite(calibration_fraction)
        and np.isfinite(threshold_fraction)
        and 0 < calibration_fraction < 1
        and 0 < threshold_fraction < 1
        and calibration_fraction + threshold_fraction < 1
    ):
        raise ValueError("Fractions must be positive and sum to less than one")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    if minimum_target_labels < 1 or minimum_dual_per_class < 1:
        raise ValueError("Minimum label/class counts must be positive")

    train = tuple(pipeline.data.train)
    validation = tuple(pipeline.data.validation)
    _validate_partition_integrity(train, validation)

    groups = np.asarray([record.scaffold_group_id for record in train])
    idx = np.arange(len(train))
    reserved_fraction = calibration_fraction + threshold_fraction
    relative_threshold_fraction = threshold_fraction / reserved_fraction

    for attempt in range(max_attempts):
        trial_seed = seed + attempt
        first = GroupShuffleSplit(
            n_splits=1,
            test_size=reserved_fraction,
            random_state=trial_seed,
        )
        fit_ix, reserved_ix = next(first.split(idx, groups=groups))
        second = GroupShuffleSplit(
            n_splits=1,
            test_size=relative_threshold_fraction,
            random_state=trial_seed + 100_003,
        )
        calibration_local_ix, threshold_local_ix = next(
            second.split(reserved_ix, groups=groups[reserved_ix])
        )
        calibration_ix = reserved_ix[calibration_local_ix]
        threshold_ix = reserved_ix[threshold_local_ix]

        # Preserve source order for reproducible feature-bank selection.
        fit = tuple(train[i] for i in sorted(fit_ix))
        calibration = tuple(train[i] for i in sorted(calibration_ix))
        threshold = tuple(train[i] for i in sorted(threshold_ix))
        _assert_disjoint_complete(train, fit, calibration, threshold)

        fit_audit = audit_calibration_records(fit)
        calibration_audit = audit_calibration_records(calibration)
        threshold_audit = audit_calibration_records(threshold)
        if not all(
            _ready(
                audit,
                minimum_target_labels=minimum_target_labels,
                minimum_dual_per_class=minimum_dual_per_class,
            )
            for audit in (fit_audit, calibration_audit, threshold_audit)
        ):
            continue

        assignments, digest = _assignments_and_hash(
            train, fit, calibration, threshold
        )
        return ScaffoldCalibrationSplit(
            fit=fit,
            calibration=calibration,
            threshold=threshold,
            fit_audit=fit_audit,
            calibration_audit=calibration_audit,
            threshold_audit=threshold_audit,
            assignments=assignments,
            assignment_sha256=digest,
            dataset_sha256=pipeline.data.dataset_sha256,
            manifest_sha256=pipeline.data.manifest_sha256,
            seed_used=trial_seed,
            requested_calibration_fraction=calibration_fraction,
            requested_threshold_fraction=threshold_fraction,
        )

    raise ValueError(
        "No valid three-way scaffold split met minimum accepted-label and "
        "dual-class requirements. Inspect training class balance or adjust "
        "fractions/minimum counts; never mix validation/test molecules into "
        "calibration."
    )
