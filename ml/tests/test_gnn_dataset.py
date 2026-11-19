"""Synthetic tests for M3.2 fit-only graph batching; no project data needed."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from dual_target.data.gnn_dataset import (
    build_fit_only_graph_bank,
    make_graph_loader,
    select_fold_graphs,
)
from dual_target.data.modeling_dataset import MoleculeRecord
from dual_target.data.pipeline import (
    FROZEN_STAGE_C_ASSIGNMENT_SHA256,
    FitOnlyData,
    FitOnlyFoldAudit,
    FitOnlyTuningPipeline,
)
from dual_target.evaluation.scaffold_cv import ScaffoldFold


def mock_pipeline() -> FitOnlyTuningPipeline:
    smiles = ("CCO", "c1ccccc1", "C1CCCCC1", "CCN", "CC(=O)O")
    records = tuple(
        MoleculeRecord(
            structure_id=f"syn_{i}", smiles=s,
            scaffold_group_id=f"scaffold_{i}",
            egfr_mask=int(i != 2), her2_mask=int(i != 1),
            egfr_pic50=None if i == 2 else 6.5 + i / 10,
            her2_pic50=None if i == 1 else 6.2 + i / 10,
        )
        for i, s in enumerate(smiles)
    )
    folds = tuple(
        ScaffoldFold(
            number=i,
            fit=tuple(row for j, row in enumerate(records) if j != i),
            holdout=(records[i],),
        )
        for i in range(5)
    )
    return FitOnlyTuningPipeline(
        data=FitOnlyData(
            train=records,
            dataset_sha256="synthetic-data-hash",
            manifest_sha256="synthetic-manifest-hash",
        ),
        folds=folds,
        fold_audits=tuple(FitOnlyFoldAudit(i, True, True, True) for i in range(5)),
        stage_c_assignment_sha256=FROZEN_STAGE_C_ASSIGNMENT_SHA256,
        stage_c_seed=719,
    )


def test_graph_bank_preserves_order_identity_and_masks() -> None:
    pipeline = mock_pipeline()
    bank = build_fit_only_graph_bank(pipeline)
    assert bank.molecules == 5
    assert bank.ordered_ids == tuple(row.structure_id for row in pipeline.data.train)
    assert len(bank.graphs_by_id) == 5
    for row in pipeline.data.train:
        graph = bank.graphs_by_id[row.structure_id]
        assert graph.structure_id == row.structure_id
        assert graph.scaffold_group_id == row.scaffold_group_id
        assert graph.y_mask.tolist() == [[bool(row.egfr_mask), bool(row.her2_mask)]]


def test_fold_graphs_batch_variable_sized_molecules() -> None:
    pipeline = mock_pipeline()
    bank = build_fit_only_graph_bank(pipeline)
    fold = select_fold_graphs(pipeline, bank, 0)
    assert len(fold.fit) == 4 and len(fold.holdout) == 1
    batch = next(iter(make_graph_loader(fold.fit, batch_size=4)))
    assert batch.num_graphs == 4
    assert batch.y.shape == (4, 2)
    assert batch.y_mask.shape == (4, 2)
    assert batch.x.shape[1] == 9
    assert batch.edge_attr.shape[1] == 4
    assert batch.structure_id == [r.structure_id for r in pipeline.folds[0].fit]
    assert len(batch.batch) == batch.x.shape[0]
    for left, right in batch.edge_index.T.tolist():
        assert batch.batch[left].item() == batch.batch[right].item()
    assert torch.equal(batch.y_mask[0], torch.tensor([True, False]))


def test_fold_holdouts_are_independent_and_cover_all_graphs() -> None:
    pipeline = mock_pipeline()
    bank = build_fit_only_graph_bank(pipeline)
    held_out = [
        graph.structure_id
        for fold in range(5)
        for graph in select_fold_graphs(pipeline, bank, fold).holdout
    ]
    assert set(held_out) == set(bank.ordered_ids)
    assert len(held_out) == len(set(held_out))


def test_changed_stage_c_hash_is_rejected() -> None:
    pipeline = replace(mock_pipeline(), stage_c_assignment_sha256="changed")
    with pytest.raises(ValueError, match="assignment hash"):
        build_fit_only_graph_bank(pipeline)


def test_leaking_scaffold_fold_is_rejected() -> None:
    pipeline = mock_pipeline()
    original = pipeline.folds[0]
    altered = replace(original, fit=original.fit + original.holdout)
    pipeline = replace(pipeline, folds=(altered, *pipeline.folds[1:]))
    with pytest.raises(ValueError, match="does not partition"):
        build_fit_only_graph_bank(pipeline)


def test_unknown_or_changed_record_is_rejected() -> None:
    pipeline = mock_pipeline()
    bank = build_fit_only_graph_bank(pipeline)
    with pytest.raises(ValueError, match="Unknown scaffold CV fold"):
        select_fold_graphs(pipeline, bank, 99)
    changed = replace(pipeline.data.train[0], egfr_pic50=1.0)
    altered_fold = replace(
        pipeline.folds[0], holdout=(changed,)
    )
    altered_pipeline = replace(
        pipeline, folds=(altered_fold, *pipeline.folds[1:])
    )
    with pytest.raises(ValueError, match="differs from frozen"):
        select_fold_graphs(altered_pipeline, bank, 0)


def test_graph_loader_checks_batch_size_and_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        make_graph_loader([])
    graph = next(iter(build_fit_only_graph_bank(mock_pipeline()).graphs_by_id.values()))
    with pytest.raises(ValueError, match="batch_size"):
        make_graph_loader([graph], batch_size=0)
