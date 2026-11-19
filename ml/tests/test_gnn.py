"""Synthetic M3.3 GNN tests: never load calibrated or held-out molecules."""

from __future__ import annotations

import pytest
import torch
from dual_target.data.modeling_dataset import MoleculeRecord
from dual_target.features.molecular_graph import (
    ATOM_CARDINALITIES,
    molecule_to_graph,
)
from dual_target.models.gnn import (
    CategoricalFeatureEncoder,
    MolecularGNN,
    masked_huber_loss,
)
from torch_geometric.loader import DataLoader


def example_graphs():
    cases = (
        ("ethanol", "CCO", 1, 0, 7.1, None),
        ("benzene", "c1ccccc1", 0, 1, None, 6.8),
        ("acetamide", "CC(=O)N", 1, 1, 7.0, 6.5),
    )
    return tuple(
        molecule_to_graph(
            MoleculeRecord(
                structure_id=sid,
                smiles=smiles,
                scaffold_group_id=f"test_scaffold_{sid}",
                egfr_mask=em,
                her2_mask=hm,
                egfr_pic50=ey,
                her2_pic50=hy,
            )
        )
        for sid, smiles, em, hm, ey, hy in cases
    )


def test_forward_is_graph_level_regression() -> None:
    model = MolecularGNN(hidden_dim=32, num_layers=2, dropout=0.0)
    batch = next(iter(DataLoader(list(example_graphs()), batch_size=3)))
    predictions = model(batch)
    assert predictions.shape == (3, 2)
    assert torch.isfinite(predictions).all()
    assert batch.y.shape == batch.y_mask.shape == (3, 2)


def test_single_molecule_without_batch_vector() -> None:
    graph = example_graphs()[0]
    model = MolecularGNN(hidden_dim=16, num_layers=1)
    assert model(graph).shape == (1, 2)


def test_loss_balances_target_means() -> None:
    predictions = torch.tensor([[6.5, 6.2], [7.0, 7.0]])
    targets = torch.tensor([[7.0, 0.0], [7.5, 6.0]])
    mask = torch.tensor([[True, False], [True, True]])
    # Huber delta=1: EGFR errors 0.5 -> .125; HER2 error 1.0 -> .5
    assert torch.isclose(masked_huber_loss(predictions, targets, mask), torch.tensor(0.3125))


def test_missing_labels_have_zero_gradient() -> None:
    predictions = torch.tensor([[6.5, 6.2], [7.0, 7.0]], requires_grad=True)
    targets = torch.tensor([[7.0, 0.0], [7.5, 6.0]])
    mask = torch.tensor([[True, False], [True, True]])
    masked_huber_loss(predictions, targets, mask).backward()
    assert predictions.grad is not None
    assert predictions.grad[0, 1].item() == 0.0
    assert predictions.grad[0, 0].item() != 0.0
    assert predictions.grad[1, 1].item() != 0.0


def test_no_her2_labels_in_batch_is_supported() -> None:
    predictions = torch.tensor([[7.5, 3.0], [7.0, 20.0]])
    targets = torch.tensor([[7.0, 0.0], [7.0, 0.0]])
    mask = torch.tensor([[True, False], [True, False]])
    assert torch.isclose(masked_huber_loss(predictions, targets, mask), torch.tensor(0.0625))


def test_all_missing_batch_fails_explicitly() -> None:
    with pytest.raises(ValueError, match="At least one observed"):
        masked_huber_loss(torch.zeros(2, 2), torch.zeros(2, 2), torch.zeros(2, 2, dtype=torch.bool))


def test_backward_reaches_atom_and_bond_embeddings_and_heads() -> None:
    model = MolecularGNN(hidden_dim=32, num_layers=2, dropout=0.0)
    batch = next(iter(DataLoader(list(example_graphs()), batch_size=3)))
    predictions = model(batch)
    loss = masked_huber_loss(predictions, batch.y, batch.y_mask)
    loss.backward()
    assert model.atom_encoder.embeddings[0].weight.grad is not None
    assert model.bond_encoder.embeddings[0].weight.grad is not None
    assert model.egfr_head[-1].weight.grad is not None
    assert model.her2_head[-1].weight.grad is not None


def test_categorical_encoder_rejects_unknown_category() -> None:
    encoder = CategoricalFeatureEncoder(ATOM_CARDINALITIES, hidden_dim=16)
    features = example_graphs()[0].x.clone()
    features[0, 0] = ATOM_CARDINALITIES[0]
    with pytest.raises(ValueError, match="out-of-range"):
        encoder(features)


def test_bad_loss_shapes_and_masks_fail() -> None:
    y = torch.zeros(3, 2)
    with pytest.raises(ValueError, match="must all be"):
        masked_huber_loss(y, y, torch.zeros(3, 1, dtype=torch.bool))
    with pytest.raises(TypeError, match="boolean"):
        masked_huber_loss(y, y, torch.ones(3, 2))


def test_eval_mode_is_repeatable() -> None:
    model = MolecularGNN(hidden_dim=16, num_layers=2, dropout=0.2).eval()
    batch = next(iter(DataLoader(list(example_graphs()), batch_size=3)))
    with torch.no_grad():
        first = model(batch)
        second = model(batch)
    torch.testing.assert_close(first, second)


def test_model_hyperparameters_rejected_when_invalid() -> None:
    with pytest.raises(ValueError, match="num_layers"):
        MolecularGNN(num_layers=0)
    with pytest.raises(ValueError, match="dropout"):
        MolecularGNN(dropout=1.0)
