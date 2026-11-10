
"""Create reusable scaffold-disjoint cross-validation folds from M2.1 training data.

Each training molecule is assigned to exactly one internal holdout fold. Within
each fold, all molecules from a scaffold group are kept together: they are
either all in the model-fitting subset or all in the holdout subset. The folds
can be reused by the EGFR/HER2 regressors and the direct dual-activity
classifier.

This module accepts the existing :class:`ModelingData` object, reusing the
project's dataset-loading and curation pipeline rather than duplicating file
loading or scaffold logic. It creates assignments only; per-fold target-label
counts and class balance are left to downstream evaluation before optimization.
The existing validation and test partitions are not used to create folds.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import GroupKFold

from ..data.modeling_dataset import ModelingData, MoleculeRecord


@dataclass(frozen=True)
class ScaffoldFold:
    """One training-only fold with disjoint fit and holdout scaffolds."""

    number: int
    fit: tuple[MoleculeRecord, ...]
    holdout: tuple[MoleculeRecord, ...]


@dataclass(frozen=True)
class FoldAssignment:
    """One molecule's internal holdout fold, for a reproducible manifest."""

    structure_id: str
    scaffold_group_id: str
    holdout_fold: int


def make_scaffold_folds(
    data: ModelingData,
    n_splits: int = 5,
) -> tuple[ScaffoldFold, ...]:
    """Build deterministic grouped folds using only the M2.1 train records.

    All records sharing a scaffold stay in the same holdout fold. The output
    can feed either target-specific RF training or paired classification.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    records = data.train
    ids = [record.structure_id for record in records]
    groups = [record.scaffold_group_id for record in records]

    if not records:
        raise ValueError("No training records")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate training molecule IDs")
    if any(not group for group in groups):
        raise ValueError("Missing training scaffold ID")
    if len(set(groups)) < n_splits:
        raise ValueError("Not enough distinct training scaffolds for folds")

    validation_ids = {record.structure_id for record in data.validation}
    validation_scaffolds = {
        record.scaffold_group_id for record in data.validation
    }
    if set(ids) & validation_ids:
        raise ValueError("Training/validation molecule overlap")
    if set(groups) & validation_scaffolds:
        raise ValueError("Training/validation scaffold overlap")

    # GroupKFold without shuffling is deterministic for a fixed frozen table.
    splitter = GroupKFold(n_splits=n_splits)
    row_indices = np.arange(len(records))
    folds = []
    seen_holdout_ids = set()

    for number, (fit_indices, holdout_indices) in enumerate(
        splitter.split(row_indices, groups=groups)
    ):
        fit = tuple(records[int(i)] for i in fit_indices)
        holdout = tuple(records[int(i)] for i in holdout_indices)

        fit_scaffolds = {record.scaffold_group_id for record in fit}
        holdout_scaffolds = {
            record.scaffold_group_id for record in holdout
        }
        if fit_scaffolds & holdout_scaffolds:
            raise ValueError("Internal scaffold leakage")

        holdout_ids = {record.structure_id for record in holdout}
        if seen_holdout_ids & holdout_ids:
            raise ValueError("A molecule occurs in multiple holdout folds")
        seen_holdout_ids.update(holdout_ids)
        folds.append(
            ScaffoldFold(number=number, fit=fit, holdout=holdout)
        )

    if seen_holdout_ids != set(ids):
        raise ValueError(
            "Some training molecules lack an internal holdout fold"
        )

    return tuple(folds)


def fold_manifest(
    folds: Sequence[ScaffoldFold],
) -> tuple[FoldAssignment, ...]:
    """Return stable molecule-to-fold assignments for CSV/MLflow logging."""
    assignments = [
        FoldAssignment(
            structure_id=record.structure_id,
            scaffold_group_id=record.scaffold_group_id,
            holdout_fold=fold.number,
        )
        for fold in folds
        for record in fold.holdout
    ]
    ids = [assignment.structure_id for assignment in assignments]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate molecule in fold manifest")
    return tuple(
        sorted(assignments, key=lambda item: item.structure_id)
    )
