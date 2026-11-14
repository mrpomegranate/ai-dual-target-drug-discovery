
"""Tests for scaffold-disjoint folds within the frozen training partition."""

from dataclasses import replace

import pytest
from dual_target.data.modeling_dataset import ModelingData, MoleculeRecord
from dual_target.evaluation.scaffold_cv import (
    ScaffoldFold,
    fold_manifest,
    make_scaffold_folds,
)


@pytest.fixture
def modeling_data():
    """Six scaffold groups with two synthetic molecules each."""
    train = tuple(
        MoleculeRecord(
            structure_id=f"molecule_{group}_{index}",
            smiles=f"{'C' * (group + 2)}{'O' if index == 0 else 'N'}",
            scaffold_group_id=f"scaffold_{group}",
            egfr_mask=int(index == 0),
            her2_mask=int(index == 1),
            egfr_pic50=7.0 if index == 0 else None,
            her2_pic50=6.5 if index == 1 else None,
        )
        for group in range(6)
        for index in range(2)
    )
    validation = (
        MoleculeRecord(
            structure_id="validation_molecule",
            smiles="CCC",
            scaffold_group_id="validation_scaffold",
            egfr_mask=1,
            her2_mask=1,
            egfr_pic50=7.5,
            her2_pic50=6.8,
        ),
    )
    return ModelingData(
        train=train,
        validation=validation,
        dataset_sha256="synthetic-dataset",
        manifest_sha256="synthetic-manifest",
    )


def test_number_of_folds_and_coverage(modeling_data):
    folds = make_scaffold_folds(modeling_data, n_splits=3)

    assert len(folds) == 3
    assert [fold.number for fold in folds] == [0, 1, 2]
    assert sum(len(fold.holdout) for fold in folds) == len(modeling_data.train)
    assert all(fold.fit and fold.holdout for fold in folds)


def test_fit_and_holdout_are_disjoint(modeling_data):
    for fold in make_scaffold_folds(modeling_data, n_splits=3):
        fit_ids = {record.structure_id for record in fold.fit}
        holdout_ids = {record.structure_id for record in fold.holdout}
        fit_groups = {record.scaffold_group_id for record in fold.fit}
        holdout_groups = {record.scaffold_group_id for record in fold.holdout}

        assert fit_ids.isdisjoint(holdout_ids)
        assert fit_groups.isdisjoint(holdout_groups)
        assert len(fit_ids | holdout_ids) == len(modeling_data.train)


def test_each_molecule_held_out_once(modeling_data):
    folds = make_scaffold_folds(modeling_data, n_splits=3)
    held_out = [record.structure_id for fold in folds for record in fold.holdout]

    assert len(held_out) == len(set(held_out))
    assert set(held_out) == {record.structure_id for record in modeling_data.train}


def test_same_scaffold_kept_in_one_holdout_fold(modeling_data):
    assignments = fold_manifest(make_scaffold_folds(modeling_data, n_splits=3))
    groups = {}
    for item in assignments:
        groups.setdefault(item.scaffold_group_id, set()).add(item.holdout_fold)

    assert len(groups) == 6
    assert all(len(fold_numbers) == 1 for fold_numbers in groups.values())


def test_manifest_is_sorted_and_deterministic(modeling_data):
    first = fold_manifest(make_scaffold_folds(modeling_data, n_splits=3))
    second = fold_manifest(make_scaffold_folds(modeling_data, n_splits=3))

    assert first == second
    assert [row.structure_id for row in first] == sorted(
        row.structure_id for row in first
    )
    assert len(first) == 12


def test_preserves_original_records_and_masks(modeling_data):
    folds = make_scaffold_folds(modeling_data, n_splits=3)
    original = {record.structure_id: record for record in modeling_data.train}

    for fold in folds:
        for record in fold.holdout:
            assert record == original[record.structure_id]


def test_original_validation_is_not_in_internal_folds(modeling_data):
    folds = make_scaffold_folds(modeling_data, n_splits=3)
    validation_ids = {record.structure_id for record in modeling_data.validation}

    for fold in folds:
        all_ids = {record.structure_id for record in fold.fit + fold.holdout}
        assert all_ids.isdisjoint(validation_ids)


@pytest.mark.parametrize("n_splits", [0, 1])
def test_rejects_too_few_folds(modeling_data, n_splits):
    with pytest.raises(ValueError, match="at least 2"):
        make_scaffold_folds(modeling_data, n_splits=n_splits)


def test_rejects_empty_training_set(modeling_data):
    data = replace(modeling_data, train=())
    with pytest.raises(ValueError, match="No training records"):
        make_scaffold_folds(data, n_splits=3)


def test_rejects_duplicate_training_molecule(modeling_data):
    data = replace(modeling_data, train=modeling_data.train + (modeling_data.train[0],))
    with pytest.raises(ValueError, match="Duplicate training molecule IDs"):
        make_scaffold_folds(data, n_splits=3)


def test_rejects_missing_scaffold_id(modeling_data):
    modified = replace(modeling_data.train[0], scaffold_group_id="")
    data = replace(modeling_data, train=(modified,) + modeling_data.train[1:])
    with pytest.raises(ValueError, match="Missing training scaffold ID"):
        make_scaffold_folds(data, n_splits=3)


def test_rejects_more_folds_than_scaffolds(modeling_data):
    with pytest.raises(ValueError, match="Not enough distinct"):
        make_scaffold_folds(modeling_data, n_splits=7)


def test_rejects_validation_molecule_overlap(modeling_data):
    modified = replace(
        modeling_data.validation[0],
        structure_id=modeling_data.train[0].structure_id,
    )
    data = replace(modeling_data, validation=(modified,))
    with pytest.raises(ValueError, match="molecule overlap"):
        make_scaffold_folds(data, n_splits=3)


def test_rejects_validation_scaffold_overlap(modeling_data):
    modified = replace(
        modeling_data.validation[0],
        scaffold_group_id=modeling_data.train[0].scaffold_group_id,
    )
    data = replace(modeling_data, validation=(modified,))
    with pytest.raises(ValueError, match="scaffold overlap"):
        make_scaffold_folds(data, n_splits=3)


def test_manifest_rejects_repeated_holdout_molecule(modeling_data):
    record = modeling_data.train[0]
    folds = (
        ScaffoldFold(number=0, fit=(), holdout=(record,)),
        ScaffoldFold(number=1, fit=(), holdout=(record,)),
    )
    with pytest.raises(ValueError, match="Duplicate molecule"):
        fold_manifest(folds)
