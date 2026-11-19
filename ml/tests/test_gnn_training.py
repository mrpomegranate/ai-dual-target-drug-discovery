"""Synthetic inner-scaffold stopping tests; no curated or reserved labels."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from dual_target.data.gnn_dataset import FoldGraphViews
from dual_target.data.modeling_dataset import MoleculeRecord
from dual_target.features.molecular_graph import molecule_to_graph
from dual_target.training.gnn_training import (
    GNNTrainingConfig,
    evaluate_potency_mae,
    make_inner_scaffold_split,
    refit_cv_training_graphs,
    train_with_inner_stopping,
)
from torch import nn


def synthetic_fit_graphs():
    """Distinct synthetic scaffold identities; all possess both target labels."""
    smiles = (
        "CCO", "CCN", "CCC", "CC(=O)O", "CC(=O)N", "CCCl",
        "c1ccccc1", "C1CCCCC1", "CCBr", "CCS", "COC", "CNC",
    )
    return tuple(
        molecule_to_graph(MoleculeRecord(
            structure_id=f"synthetic_{i}",
            smiles=s,
            scaffold_group_id=f"synthetic_scaffold_{i}",
            egfr_mask=1,
            her2_mask=1,
            egfr_pic50=6.0 + 0.1 * i,
            her2_pic50=7.0 - 0.05 * i,
        ))
        for i, s in enumerate(smiles)
    )


def view(graphs):
    return FoldGraphViews(number=0, fit=graphs, holdout=(
        molecule_to_graph(MoleculeRecord(
            structure_id="cv_holdout_sentinel", smiles="c1ccncc1",
            scaffold_group_id="cv_holdout_scaffold",
            egfr_mask=1, her2_mask=1,
            egfr_pic50=8.0, her2_pic50=8.0,
        )),
    ))


def tiny_config(**changes):
    return replace(
        GNNTrainingConfig(
            hidden_dim=16, num_layers=1, dropout=0.0,
            batch_size=5, max_epochs=2, patience=2,
        ), **changes
    )


def test_inner_scaffold_split_is_deterministic_and_complete():
    graphs = synthetic_fit_graphs()
    first = make_inner_scaffold_split(graphs, seed=719, fraction=0.25)
    second = make_inner_scaffold_split(graphs, seed=719, fraction=0.25)
    first_train_ids = {g.structure_id for g in first.train}
    first_stop_ids = {g.structure_id for g in first.stop}
    assert first_train_ids == {g.structure_id for g in second.train}
    assert first_stop_ids == {g.structure_id for g in second.stop}
    assert first_train_ids.isdisjoint(first_stop_ids)
    assert first_train_ids | first_stop_ids == {g.structure_id for g in graphs}
    assert {g.scaffold_group_id for g in first.train}.isdisjoint(
        {g.scaffold_group_id for g in first.stop}
    )


def test_inner_split_does_not_consult_an_external_cv_holdout():
    # These are precisely the CV *fit* graphs. The sentinel is never supplied.
    cv_fit = synthetic_fit_graphs()[:-1]
    sentinel = synthetic_fit_graphs()[-1].structure_id
    partition = make_inner_scaffold_split(cv_fit, seed=719)
    assert sentinel not in {g.structure_id for g in (*partition.train, *partition.stop)}


def test_split_rejects_duplicate_molecule_ids_and_bad_fraction():
    graphs = synthetic_fit_graphs()
    with pytest.raises(ValueError, match="Duplicate molecule"):
        make_inner_scaffold_split((*graphs, graphs[0]))
    with pytest.raises(ValueError, match="fraction"):
        make_inner_scaffold_split(graphs, fraction=1.0)


def test_stop_subset_requires_both_targets():
    graphs = tuple(synthetic_fit_graphs()[:4])
    for graph in graphs:
        graph.y_mask[0, 1] = False
        graph.y[0, 1] = 0.0
    with pytest.raises(ValueError, match="Both EGFR and HER2"):
        make_inner_scaffold_split(graphs, fraction=0.5)


def test_evaluate_mae_aggregates_measured_labels_only():
    graphs = synthetic_fit_graphs()[:3]
    # One missing HER2 label must NOT be an observed zero.
    graphs[0].y_mask[0, 1] = False
    graphs[0].y[0, 1] = 0.0

    class ConstantModel(nn.Module):
        def forward(self, batch):
            return torch.tensor([[6.5, 6.5]]).repeat(batch.y.size(0), 1)

    metrics = evaluate_potency_mae(ConstantModel(), graphs, batch_size=2, device="cpu")
    assert (metrics.egfr_count, metrics.her2_count) == (3, 2)
    assert metrics.egfr_mae == pytest.approx((0.5 + 0.4 + 0.3) / 3, abs=1e-5)
    assert metrics.her2_mae == pytest.approx((0.45 + 0.4) / 2, abs=1e-5)
    assert metrics.balanced_mae == pytest.approx(
        (metrics.egfr_mae + metrics.her2_mae) / 2
    )


def test_one_short_inner_stopping_run_and_fresh_refit():
    graphs = synthetic_fit_graphs()
    result = train_with_inner_stopping(view(graphs), config=tiny_config(), device="cpu")
    assert 1 <= result.best_epoch <= result.epochs_ran <= 2
    assert result.inner_train_molecules + result.inner_stop_molecules == len(graphs)
    assert result.inner_train_scaffolds + result.inner_stop_scaffolds == len(graphs)
    assert len(result.history) == result.epochs_ran
    assert result.history[0].stopping_metrics.egfr_count > 0
    assert result.history[0].stopping_metrics.her2_count > 0
    assert torch.isfinite(torch.tensor(result.best_inner_mae))
    fresh = refit_cv_training_graphs(
        view(graphs), epochs=result.best_epoch, config=tiny_config(), device="cpu"
    )
    assert fresh.training is False
    assert all(torch.isfinite(parameter).all() for parameter in fresh.parameters())


def test_invalid_epoch_or_config_is_rejected():
    graphs = synthetic_fit_graphs()
    with pytest.raises(ValueError, match="epochs"):
        refit_cv_training_graphs(view(graphs), epochs=0, config=tiny_config())
    with pytest.raises(ValueError, match="learning_rate"):
        GNNTrainingConfig(learning_rate=0.0)
    with pytest.raises(ValueError, match="batch_size"):
        GNNTrainingConfig(batch_size=0)


def test_training_requires_fold_view_and_rejects_cv_holdout_leakage():
    graphs = synthetic_fit_graphs()
    with pytest.raises(TypeError, match="FoldGraphViews"):
        train_with_inner_stopping(graphs, config=tiny_config())
    bad_view = FoldGraphViews(
        number=0, fit=graphs, holdout=(graphs[0],),
    )
    with pytest.raises(ValueError, match="CV fold molecule leakage"):
        train_with_inner_stopping(bad_view, config=tiny_config())
