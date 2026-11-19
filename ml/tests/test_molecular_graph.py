"""Focused tests for molecular graph conversion; synthetic labels only."""

from __future__ import annotations

import pytest
import torch
from dual_target.data.modeling_dataset import MoleculeRecord
from dual_target.features.molecular_graph import (
    ATOM_CARDINALITIES,
    ATOM_FEATURE_NAMES,
    BOND_CARDINALITIES,
    BOND_FEATURE_NAMES,
    GRAPH_SCHEMA_VERSION,
    molecule_to_graph,
)
from torch_geometric.loader import DataLoader


def record(
    smiles: str = "CCO", *, egfr: float | None = 7.2,
    her2: float | None = None, sid: str = "synthetic_001",
) -> MoleculeRecord:
    return MoleculeRecord(
        structure_id=sid,
        smiles=smiles,
        scaffold_group_id=f"scaffold_{sid}",
        egfr_mask=int(egfr is not None),
        her2_mask=int(her2 is not None),
        egfr_pic50=egfr,
        her2_pic50=her2,
    )


def test_ethanol_shape_edges_and_missing_mask() -> None:
    graph = molecule_to_graph(record())
    assert graph.x.shape == (3, len(ATOM_FEATURE_NAMES))
    assert graph.edge_index.shape == (2, 4)
    assert graph.edge_attr.shape == (4, len(BOND_FEATURE_NAMES))
    assert graph.y.shape == graph.y_mask.shape == (1, 2)
    assert graph.y_mask.tolist() == [[True, False]]
    assert graph.y.tolist()[0] == pytest.approx([7.2, 0.0])
    assert graph.structure_id == "synthetic_001"
    assert graph.graph_schema_version == GRAPH_SCHEMA_VERSION
    connections = set(map(tuple, graph.edge_index.t().tolist()))
    assert connections == {(0, 1), (1, 0), (1, 2), (2, 1)}
    assert torch.equal(graph.edge_attr[0], graph.edge_attr[1])
    assert torch.equal(graph.edge_attr[2], graph.edge_attr[3])


def test_feature_vocabulary_indices_are_in_range() -> None:
    for smiles in ("CCO", "c1ccccc1", "F[C@H](Cl)Br", "F/C=C/F", "[Na+]"):
        graph = molecule_to_graph(record(smiles))
        for col, size in enumerate(ATOM_CARDINALITIES):
            assert torch.all((0 <= graph.x[:, col]) & (graph.x[:, col] < size))
        for col, size in enumerate(BOND_CARDINALITIES):
            assert torch.all(
                (0 <= graph.edge_attr[:, col]) & (graph.edge_attr[:, col] < size)
            )


def test_stereochemical_isomers_remain_distinguishable() -> None:
    r = molecule_to_graph(record("F[C@H](Cl)Br"))
    s = molecule_to_graph(record("F[C@@H](Cl)Br"))
    assert not torch.equal(r.x, s.x)
    e = molecule_to_graph(record("F/C=C/F"))
    z = molecule_to_graph(record("F/C=C\\F"))
    assert not torch.equal(e.edge_attr, z.edge_attr)


def test_single_atom_graph_has_empty_edge_matrix() -> None:
    graph = molecule_to_graph(record("[Na+]"))
    assert graph.x.shape[0] == 1
    assert graph.edge_index.shape == (2, 0)
    assert graph.edge_attr.shape == (0, len(BOND_FEATURE_NAMES))


def test_batch_keeps_graph_targets_and_identities() -> None:
    dataset = [
        molecule_to_graph(record("CCO", egfr=7.2, sid="A")),
        molecule_to_graph(record("C1CCCCC1", egfr=None, her2=6.1, sid="B")),
    ]
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
    assert batch.x.shape[0] == 9
    assert batch.y.shape == (2, 2)
    assert batch.y_mask.tolist() == [[True, False], [False, True]]
    assert batch.batch.tolist() == [0] * 3 + [1] * 6
    assert batch.structure_id == ["A", "B"]
    assert batch.scaffold_group_id == ["scaffold_A", "scaffold_B"]
    # The first graph and second graph stay disconnected in the batch.
    for i, j in batch.edge_index.t().tolist():
        assert batch.batch[i] == batch.batch[j]


def test_deterministic_graph_features() -> None:
    a = molecule_to_graph(record("c1ccccc1"))
    b = molecule_to_graph(record("c1ccccc1"))
    for field in ("x", "edge_index", "edge_attr", "y", "y_mask"):
        assert torch.equal(getattr(a, field), getattr(b, field))


@pytest.mark.parametrize("smiles", ["", "not_a_smiles"])
def test_invalid_smiles_fail_loudly(smiles: str) -> None:
    with pytest.raises(ValueError, match="SMILES"):
        molecule_to_graph(record(smiles))


def test_no_accepted_labels_fails() -> None:
    with pytest.raises(ValueError, match="no accepted target"):
        molecule_to_graph(record(egfr=None, her2=None))


def test_missing_label_is_not_automatically_inactive() -> None:
    graph = molecule_to_graph(record(egfr=None, her2=8.4))
    assert graph.y_mask.tolist() == [[False, True]]
    assert graph.y.tolist()[0] == pytest.approx([0.0, 8.4])


def test_nonfinite_accepted_label_fails() -> None:
    with pytest.raises(ValueError, match="label invalid"):
        molecule_to_graph(record(egfr=float("nan")))
