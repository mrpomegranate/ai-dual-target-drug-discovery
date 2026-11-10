
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
