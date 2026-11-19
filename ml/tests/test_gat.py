"""M3.6 edge-aware GATv2 tests with synthetic molecules only."""

from __future__ import annotations

import pytest
import torch
from dual_target.data.modeling_dataset import MoleculeRecord
from dual_target.features.molecular_graph import molecule_to_graph
from dual_target.models.gat import MolecularGATv2
from dual_target.models.gnn import masked_huber_loss
from torch_geometric.loader import DataLoader


def example_graphs():
    examples = (
        ("ethanol", "CCO", 1, 0, 7.1, None),
        ("benzene", "c1ccccc1", 0, 1, None, 6.8),
        ("acetamide", "CC(=O)N", 1, 1, 7.0, 6.5),
    )
    return tuple(
        molecule_to_graph(
            MoleculeRecord(
                structure_id=sid,
                smiles=smiles,
                scaffold_group_id=f"fixture_{sid}",
                egfr_mask=em,
                her2_mask=hm,
                egfr_pic50=ey,
                her2_pic50=hy,
            )
        )
        for sid, smiles, em, hm, ey, hy in examples
    )


def test_gatv2_is_graph_level_multitask_regression() -> None:
    torch.manual_seed(719)
    batch = next(iter(DataLoader(list(example_graphs()), batch_size=3)))
    model = MolecularGATv2(hidden_dim=32, num_layers=2, heads=4, dropout=0.0)
    predicted = model(batch)
    assert predicted.shape == (3, 2)
    assert torch.isfinite(predicted).all()
    assert batch.y.shape == batch.y_mask.shape == (3, 2)


def test_single_molecule_without_batch_vector() -> None:
    graph = example_graphs()[0]
    assert MolecularGATv2(hidden_dim=16, heads=4, num_layers=1)(graph).shape == (1, 2)


def test_attention_layers_use_bonds_and_preserve_hidden_width() -> None:
    model = MolecularGATv2(hidden_dim=128, num_layers=3, heads=4)
    assert len(model.convs) == 3
    for conv in model.convs:
        assert conv.heads == 4
        assert conv.out_channels == 32
        assert conv.edge_dim == 128
        assert conv.concat
        assert conv.add_self_loops


def test_gradient_reaches_bond_attention_and_both_heads() -> None:
    torch.manual_seed(719)
    model = MolecularGATv2(hidden_dim=32, num_layers=2, heads=4, dropout=0.0)
    batch = next(iter(DataLoader(list(example_graphs()), batch_size=3)))
    loss = masked_huber_loss(model(batch), batch.y, batch.y_mask)
    loss.backward()
    assert model.atom_encoder.embeddings[0].weight.grad is not None
    bond_grad = model.bond_encoder.embeddings[0].weight.grad
    assert bond_grad is not None and torch.isfinite(bond_grad).all()
    assert bond_grad.abs().sum().item() > 0
    assert model.egfr_head[-1].weight.grad is not None
    assert model.her2_head[-1].weight.grad is not None


def test_masked_labels_have_no_loss_gradient() -> None:
    predictions = torch.tensor([[6.5, 6.2], [7.0, 7.0]], requires_grad=True)
    targets = torch.tensor([[7.0, 0.0], [7.5, 6.0]])
    masks = torch.tensor([[True, False], [True, True]])
    masked_huber_loss(predictions, targets, masks).backward()
    assert predictions.grad is not None
    assert predictions.grad[0, 1] == 0


def test_eval_mode_is_deterministic() -> None:
    torch.manual_seed(719)
    model = MolecularGATv2(hidden_dim=32, heads=4, attention_dropout=0.2).eval()
    batch = next(iter(DataLoader(list(example_graphs()), batch_size=3)))
    with torch.no_grad():
        first, second = model(batch), model(batch)
    torch.testing.assert_close(first, second)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"hidden_dim": 30, "heads": 4}, "divisible"),
        ({"num_layers": 0}, "num_layers"),
        ({"heads": 0}, "heads"),
        ({"dropout": 1.0}, "dropout"),
        ({"attention_dropout": 1.0}, "attention_dropout"),
    ],
)
def test_invalid_architectures_fail(kwargs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        MolecularGATv2(**kwargs)


def test_inconsistent_bond_schema_fails() -> None:
    graph = example_graphs()[0]
    graph.edge_attr = graph.edge_attr[:, :2]
    with pytest.raises(ValueError, match="bond-feature"):
        MolecularGATv2(hidden_dim=16, heads=4)(graph)
