"""Synthetic contract tests for M1.8d collision review."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "review_structure_collisions.py"
spec = importlib.util.spec_from_file_location("review_structure_collisions", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(mod)


def data():
    return pd.DataFrame(
        [
            {"reactant_set_id": "1", "smiles": "C-C-O", "target": "EGFR", "source": "ChEMBL", "m17_partition": "primary_exact_candidate", "m18c_status": "provisional_exact_structure", "m18c_structure_id": "ID1", "m18c_candidate_isomeric_smiles": "CCO", "m18c_candidate_rdkit_inchikey": "KEY1"},
            {"reactant_set_id": "2", "smiles": "OCC", "target": "HER2", "source": "ChEMBL", "m17_partition": "primary_exact_candidate", "m18c_status": "provisional_exact_structure", "m18c_structure_id": "ID1", "m18c_candidate_isomeric_smiles": "CCO", "m18c_candidate_rdkit_inchikey": "KEY1"},
            {"reactant_set_id": "3", "smiles": "C=C-O", "target": "EGFR", "source": "US Patent", "m17_partition": "secondary_construct_reserve", "m18c_status": "provisional_exact_structure", "m18c_structure_id": "ID2", "m18c_candidate_isomeric_smiles": "C=CO", "m18c_candidate_rdkit_inchikey": "KEY1"},
            {"reactant_set_id": "4", "smiles": "bad", "target": "EGFR", "source": "US Patent", "m17_partition": "primary_exact_candidate", "m18c_status": "hold_rdkit_parse_failure", "m18c_structure_id": "", "m18c_candidate_isomeric_smiles": "", "m18c_candidate_rdkit_inchikey": ""},
        ]
    )


def test_both_collision_types():
    approved, rows = mod.collision_groups(data())
    assert len(approved) == 3
    assert len(rows) == 2
    assert {r["collision_type"] for r in rows} == {mod.TYPE_RAW, mod.TYPE_KEY}


def test_no_merging_across_inchikey():
    _, rows = mod.collision_groups(data())
    r = next(r for r in rows if r["collision_type"] == mod.TYPE_KEY)
    assert r["unique_structure_ids"] == 2
    assert r["unique_canonical_smiles"] == 2
    assert "Retain distinct structure IDs" in r["recommended_action"]


def test_only_approved_rows_considered():
    _, rows = mod.collision_groups(data())
    assert all(r["observations"] <= 3 for r in rows)


def test_missing_columns_rejected():
    with pytest.raises(ValueError, match="Missing required"):
        mod.collision_groups(data().drop(columns=["source"]))


def test_duplicate_record_rejected():
    df = data()
    df.loc[1, "reactant_set_id"] = "1"
    with pytest.raises(ValueError, match="unique reactant_set_id"):
        mod.collision_groups(df)


def test_id_conflict_rejected():
    df = data()
    df.loc[1, "m18c_structure_id"] = "OTHER"
    with pytest.raises(ValueError, match="multiple provisional structure IDs"):
        mod.collision_groups(df)


def test_empty_approved_smiles_rejected():
    df = data()
    df.loc[0, "m18c_candidate_isomeric_smiles"] = ""
    with pytest.raises(ValueError, match="missing m18c_candidate_isomeric_smiles"):
        mod.collision_groups(df)


def test_inventory_matches(tmp_path):
    _, rows = mod.collision_groups(data())
    inv = tmp_path / "collisions.csv"
    pd.DataFrame([
        {"collision_type": r["collision_type"], "identifier": r["identifier"], "distinct_structures_or_representations": r["unique_raw_smiles"] if r["collision_type"] == mod.TYPE_RAW else r["unique_canonical_smiles"]}
        for r in rows
    ]).to_csv(inv, index=False)
    mod.check_inventory(rows, inv)


def test_inventory_mismatch_rejected(tmp_path):
    _, rows = mod.collision_groups(data())
    inv = tmp_path / "collisions.csv"
    pd.DataFrame([{"collision_type": mod.TYPE_RAW, "identifier": "CCO", "distinct_structures_or_representations": 9}]).to_csv(inv, index=False)
    with pytest.raises(ValueError, match="do not match"):
        mod.check_inventory(rows, inv)
