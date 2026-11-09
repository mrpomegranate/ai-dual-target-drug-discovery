"""Tests for non-destructive M1.8a structure QA."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "audit_molecular_identity.py"
spec = importlib.util.spec_from_file_location("m18_identity_audit", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_ethanol_parses():
    q = module.profile_structure("CCO")
    assert q.status == "parsed"
    assert q.canonical_isomeric_smiles == "CCO"
    assert q.fragment_count == 1
    assert q.rdkit_inchi_key is not None


def test_invalid_smiles_flagged():
    assert module.profile_structure("C1CC").status == "rdkit_parse_failure"


def test_missing_smiles_flagged():
    assert module.profile_structure(None).status == "missing_smiles"
    assert module.profile_structure("").status == "missing_smiles"


def test_fragment_flag_does_not_discard():
    q = module.profile_structure("CCO.[Na+]")
    assert q.status == "parsed"
    assert q.fragment_count == 2
    assert "." in q.canonical_isomeric_smiles


def test_stereochemistry_preserved():
    a = module.profile_structure("N[C@@H](C)C(=O)O")
    b = module.profile_structure("N[C@H](C)C(=O)O")
    assert a.canonical_isomeric_smiles != b.canonical_isomeric_smiles
    assert a.rdkit_inchi_key != b.rdkit_inchi_key


@pytest.mark.parametrize(
    ("source", "computed", "expected"),
    [
        ("AAAA-BBBB-C", "AAAA-BBBB-C", "exact_match"),
        ("AAAA-XXXX-C", "AAAA-BBBB-C", "same_connectivity_block_only"),
        ("ZZZZ-XXXX-C", "AAAA-BBBB-C", "different_connectivity_block"),
        (None, "AAAA-BBBB-C", "source_key_missing"),
        ("AAAA-BBBB-C", None, "rdkit_key_unavailable"),
    ],
)
def test_compare_inchikey(source, computed, expected):
    assert module.compare_inchikey(source, computed) == expected


def test_no_rows_dropped_and_summary(tmp_path):
    input_path = tmp_path / "input.csv"
    out_path = tmp_path / "out.csv"
    report_path = tmp_path / "report.json"
    df = pd.DataFrame(
        [
            {
                "reactant_set_id": "1", "target": "EGFR", "smiles": "CCO",
                "inchi_key": module.profile_structure("CCO").rdkit_inchi_key,
                "m17_partition": "primary_exact_candidate", "measurement_status": "exact",
                "source": "ChEMBL",
            },
            {
                "reactant_set_id": "2", "target": "HER2", "smiles": "C1CC",
                "inchi_key": "FICTITIOUS-KEY-N", "m17_partition": "secondary_construct_reserve",
                "measurement_status": "exact", "source": "US Patent",
            },
        ]
    )
    df.to_csv(input_path, index=False)
    result = module.audit(input_path, out_path, report_path)
    output = pd.read_csv(out_path)
    assert result["observations"] == 2
    assert output["reactant_set_id"].tolist() == [1, 2]
    assert output["rdkit_status"].tolist() == ["parsed", "rdkit_parse_failure"]
    assert report_path.is_file()


def test_duplicate_reactant_id_rejected(tmp_path):
    df = pd.DataFrame(
        [
            {
                "reactant_set_id": "1",
                "target": "EGFR",
                "smiles": "CCO",
                "inchi_key": "X",
                "m17_partition": "primary_exact_candidate",
                "measurement_status": "exact",
                "source": "ChEMBL",
            }
        ] * 2
    )
    input_path = tmp_path / "input.csv"
    df.to_csv(input_path, index=False)
    with pytest.raises(ValueError, match="unique"):
        module.audit(input_path, tmp_path / "out.csv", tmp_path / "report.json")
