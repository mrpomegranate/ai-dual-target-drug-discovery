"""Representative scaffold depiction is drawn from fit-role records only."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.render_scaffold_difficulty import core_from_smiles, representative_cores


def record(structure_id, scaffold_group_id, smiles):
    return SimpleNamespace(
        structure_id=structure_id,
        scaffold_group_id=scaffold_group_id,
        smiles=smiles,
    )


def test_representation_extracts_murcko_ring_core():
    core, acyclic = core_from_smiles("Oc1ccccc1")
    assert core == "c1ccccc1"
    assert not acyclic


def test_acyclic_identity_is_not_drawn_as_empty():
    core, acyclic = core_from_smiles("CCO")
    assert core == "CCO"
    assert acyclic


def test_representative_core_uses_most_frequent_core():
    rows = [record("z", "group", "Oc1ccccc1"),
            record("a", "group", "Nc1ccccc1"),
            record("c", "group", "CCO")]
    result = representative_cores(rows, {"group"})["group"]
    assert result["fit_molecules_in_scaffold"] == 3
    assert result["representative_core_smiles"] == "c1ccccc1"
    assert result["distinct_murcko_cores_in_group"] == 2
    assert result["representative_structure_id"] == "a"


def test_missing_fit_role_scaffold_rejected():
    with pytest.raises(ValueError, match="absent"):
        representative_cores([], {"missing"})
