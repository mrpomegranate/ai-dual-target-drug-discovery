
"""Tests for the composable, label-aware M2 modeling pipeline."""

from dataclasses import replace

import pytest
from dual_target.data.modeling_dataset import ModelingData, MoleculeRecord
from dual_target.data.pipeline import (
    LabelCounts,
    compose_modeling_pipeline,
    load_modeling_pipeline,
    select_task_records,
    summarize_labels,
)


@pytest.fixture
def modeling_data():
    """Synthetic molecules across six distinct training scaffold groups."""
    cases = (
        ("paired_active", 7.4, 6.8),
        ("paired_inactive", 6.0, 7.2),
        ("egfr_only", 8.0, None),
        ("her2_only", None, 6.7),
    )
    train = tuple(
        MoleculeRecord(
            structure_id=f"g{group}_{name}",
            smiles="CCO",
            scaffold_group_id=f"scaffold_{group}",
            egfr_mask=int(egfr is not None),
            her2_mask=int(her2 is not None),
            egfr_pic50=egfr,
            her2_pic50=her2,
        )
        for group in range(6)
        for name, egfr, her2 in cases
    )
    validation = (
        MoleculeRecord("v_pair", "CCN", "validation_1", 1, 1, 7.0, 5.9),
        MoleculeRecord("v_egfr", "CCC", "validation_2", 1, 0, 8.0, None),
        MoleculeRecord("v_her2", "COC", "validation_3", 0, 1, None, 7.1),
    )
    return ModelingData(
        train=train,
        validation=validation,
        dataset_sha256="synthetic-dataset",
        manifest_sha256="synthetic-manifest",
    )


def test_training_label_counts(modeling_data):
    assert summarize_labels(modeling_data.train) == LabelCounts(
        molecules=24,
        egfr_labels=18,
        her2_labels=18,
        paired_labels=12,
        dual_active=6,
        not_dual_active=6,
    )


def test_validation_label_counts(modeling_data):
    assert summarize_labels(modeling_data.validation) == LabelCounts(
        3, 2, 2, 1, 0, 1
    )


@pytest.mark.parametrize(
    "task, expected_count",
    [("egfr", 18), ("her2", 18), ("dual", 12)],
)
def test_task_selection_respects_masks(modeling_data, task, expected_count):
    selected = select_task_records(modeling_data.train, task)

    assert len(selected) == expected_count

    if task == "egfr":
        assert all(row.egfr_mask == 1 for row in selected)
    elif task == "her2":
        assert all(row.her2_mask == 1 for row in selected)
    else:
        assert all(
            row.egfr_mask == 1 and row.her2_mask == 1 for row in selected
        )


def test_task_selection_retains_input_order(modeling_data):
    selected = select_task_records(modeling_data.train, "dual")

    assert selected == tuple(
        row
        for row in modeling_data.train
        if row.egfr_mask and row.her2_mask
    )


def test_unknown_task_is_rejected(modeling_data):
    with pytest.raises(ValueError, match="Unknown modeling task"):
        select_task_records(modeling_data.train, "unknown")


def test_activity_boundary_is_strict(modeling_data):
    boundary = next(
        row
        for row in modeling_data.train
        if "paired_inactive" in row.structure_id
    )

    assert summarize_labels((boundary,)).dual_active == 0

    more_potent = replace(boundary, egfr_pic50=6.0001)

    assert summarize_labels((more_potent,)).dual_active == 1


def test_masked_label_is_rejected(modeling_data):
    row = replace(modeling_data.train[0], egfr_mask=0)

    with pytest.raises(ValueError, match="Masked EGFR target"):
        summarize_labels((row,))


def test_missing_accepted_label_is_rejected(modeling_data):
    row = replace(modeling_data.train[0], her2_pic50=None)

    with pytest.raises(ValueError, match="Accepted HER2 label"):
        summarize_labels((row,))


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_nonfinite_accepted_label_is_rejected(modeling_data, invalid):
    row = replace(modeling_data.train[0], egfr_pic50=invalid)

    with pytest.raises(ValueError, match="nonfinite"):
        summarize_labels((row,))


def test_invalid_mask_is_rejected(modeling_data):
    row = replace(modeling_data.train[0], egfr_mask=2)

    with pytest.raises(ValueError, match="Invalid EGFR mask"):
        summarize_labels((row,))


def test_molecule_with_no_accepted_labels_is_rejected(modeling_data):
    row = replace(
        modeling_data.train[0],
        egfr_mask=0,
        her2_mask=0,
        egfr_pic50=None,
        her2_pic50=None,
    )

    with pytest.raises(ValueError, match="no accepted targets"):
        summarize_labels((row,))


def test_composed_pipeline_counts_and_folds(modeling_data):
    pipeline = compose_modeling_pipeline(modeling_data, n_splits=3)

    assert pipeline.data is modeling_data
    assert pipeline.train_counts.molecules == 24
    assert pipeline.validation_counts.molecules == 3
    assert len(pipeline.folds) == 3
    assert len(pipeline.fold_audits) == 3

    assert sum(a.holdout.molecules for a in pipeline.fold_audits) == 24
    assert sum(a.holdout.paired_labels for a in pipeline.fold_audits) == 12

    assert all(a.egfr_regression_ready for a in pipeline.fold_audits)
    assert all(a.her2_regression_ready for a in pipeline.fold_audits)
    assert all(a.dual_classification_ready for a in pipeline.fold_audits)


def test_pipeline_does_not_use_external_validation_in_folds(modeling_data):
    pipeline = compose_modeling_pipeline(modeling_data, n_splits=3)
    validation_ids = {
        row.structure_id for row in modeling_data.validation
    }

    for fold in pipeline.folds:
        fold_ids = {
            row.structure_id for row in fold.fit + fold.holdout
        }
        assert fold_ids.isdisjoint(validation_ids)


def test_pipeline_assignments_are_deterministic(modeling_data):
    first = compose_modeling_pipeline(modeling_data, n_splits=3)
    second = compose_modeling_pipeline(modeling_data, n_splits=3)

    assert first.folds == second.folds
    assert first.fold_audits == second.fold_audits


def test_pipeline_rejects_validation_scaffold_overlap(modeling_data):
    repeated = replace(
        modeling_data.validation[0],
        scaffold_group_id=modeling_data.train[0].scaffold_group_id,
    )
    changed = replace(
        modeling_data,
        validation=(repeated,) + modeling_data.validation[1:],
    )

    with pytest.raises(ValueError, match="scaffold overlap"):
        compose_modeling_pipeline(changed, n_splits=3)


def test_load_pipeline_composes_existing_loader(modeling_data, monkeypatch):
    import dual_target.data.pipeline as pipeline_module

    paths = ("dataset.csv", "manifest.csv", "summary.json")
    seen = []

    def fake_loader(*args):
        seen.extend(args)
        return modeling_data

    monkeypatch.setattr(
        pipeline_module,
        "load_modeling_data",
        fake_loader,
    )

    result = load_modeling_pipeline(*paths, n_splits=3)

    assert tuple(seen) == paths
    assert result.data is modeling_data
    assert len(result.folds) == 3
