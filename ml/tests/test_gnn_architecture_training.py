"""M3.6b synthetic architecture integration tests; no curated labels or outer CV."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from dual_target.data.gnn_dataset import FoldGraphViews
from dual_target.data.modeling_dataset import MoleculeRecord
from dual_target.features.molecular_graph import molecule_to_graph
from dual_target.models.gat import MolecularGATv2
from dual_target.models.gnn import MolecularGNN
from dual_target.training.gnn_training import (
    GNNTrainingConfig,
    _make_model,
    refit_cv_training_graphs,
    train_with_inner_stopping,
)


def _config(**overrides) -> GNNTrainingConfig:
    return replace(
        GNNTrainingConfig(
            architecture="gatv2",
            hidden_dim=16,
            num_layers=1,
            attention_heads=4,
            attention_dropout=0.0,
            dropout=0.0,
            batch_size=4,
            max_epochs=1,
            patience=1,
        ),
        **overrides,
    )


def _example_fold() -> FoldGraphViews:
    structures = (
        "CCO", "CCN", "CCC", "CCCl", "CCBr", "CCS", "COC", "CNC",
        "CC(=O)O", "CC(=O)N", "c1ccccc1", "C1CCCCC1",
    )
    graphs = tuple(
        molecule_to_graph(
            MoleculeRecord(
                structure_id=f"synthetic_{i}",
                smiles=smiles,
                scaffold_group_id=f"scaffold_{i}",
                egfr_mask=1,
                her2_mask=1,
                egfr_pic50=6.0 + 0.1 * i,
                her2_pic50=7.0 - 0.05 * i,
            )
        )
        for i, smiles in enumerate(structures)
    )
    sentinel = molecule_to_graph(
        MoleculeRecord(
            structure_id="untouched_cv_holdout",
            smiles="c1ccncc1",
            scaffold_group_id="cv_holdout_scaffold",
            egfr_mask=1,
            her2_mask=1,
            egfr_pic50=8.0,
            her2_pic50=8.0,
        )
    )
    return FoldGraphViews(number=0, fit=graphs, holdout=(sentinel,))


def test_existing_gine_is_still_default() -> None:
    config = GNNTrainingConfig(hidden_dim=16, num_layers=1)
    assert config.architecture == "gine"
    model = _make_model(config)
    assert isinstance(model, MolecularGNN)
    assert model.hidden_dim == 16


def test_factory_builds_configured_bond_aware_gatv2() -> None:
    config = _config(num_layers=2, attention_dropout=0.15)
    model = _make_model(config)
    assert isinstance(model, MolecularGATv2)
    assert len(model.convs) == 2
    assert model.convs[0].heads == 4
    assert model.convs[0].edge_dim == 16
    assert model.attention_dropout == pytest.approx(0.15)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"architecture": "gin"}, "architecture"),
        ({"architecture": "gatv2", "hidden_dim": 15}, "divisible"),
        ({"attention_heads": 0}, "attention_heads"),
        ({"attention_heads": True}, "attention_heads"),
        ({"attention_dropout": 1.0}, "attention_dropout"),
    ],
)
def test_invalid_architecture_configs_fail(kwargs, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        _config(**kwargs)


def test_gatv2_train_with_inner_scaffold_stopping_then_refit() -> None:
    fold = _example_fold()
    config = _config()
    selected = train_with_inner_stopping(fold, config=config, device="cpu")
    assert isinstance(selected.model, MolecularGATv2)
    assert selected.best_epoch == 1
    assert selected.inner_train_molecules + selected.inner_stop_molecules == 12
    assert selected.inner_train_scaffolds + selected.inner_stop_scaffolds == 12
    assert selected.inner_stop_molecules > 0
    assert torch.isfinite(torch.tensor(selected.best_inner_mae))
    refitted = refit_cv_training_graphs(
        fold, epochs=selected.best_epoch, config=config, device="cpu"
    )
    assert isinstance(refitted, MolecularGATv2)
    assert not refitted.training
    assert all(torch.isfinite(p).all() for p in refitted.parameters())
