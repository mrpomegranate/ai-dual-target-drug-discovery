"""Unit and integration tests for conservative M1.8c structure identity."""

import csv
import json

import pytest
from dual_target.standardization import (
    classify_structure,
    structure_id,
    unspecified_stereogenic_elements,
)


def qa(**kwargs):
    return {
        "smiles": "CCO",
        "rdkit_status": "parsed",
        "source_key_comparison": "exact_match",
        "rdkit_fragment_count": "1.0",
        "rdkit_canonical_isomeric_smiles": "CCO",
        **kwargs,
    }


def test_structure_id_is_deterministic():
    assert structure_id("CCO") == structure_id("CCO")


def test_structure_id_distinguishes_different_structures():
    assert structure_id("CCO") != structure_id("CCN")


def test_structure_id_distinguishes_specified_stereoisomers():
    assert structure_id("N[C@H](C)O") != structure_id("N[C@@H](C)O")


def test_structure_id_rejects_blank():
    with pytest.raises(ValueError):
        structure_id("")


def test_good_structure_provisional():
    assert classify_structure(qa(), set()) == "provisional_exact_structure"


def test_partial_inchikey_match_not_automatically_rejected():
    result = qa(source_key_comparison="same_connectivity_block_only")
    assert classify_structure(result, set()) == "provisional_exact_structure"


def test_flagged_raw_smiles_hold():
    assert classify_structure(qa(), {"CCO"}) == "hold_inchikey_connectivity_disagreement"


def test_connectivity_disagreement_hold_even_without_priority_list():
    result = qa(source_key_comparison="different_connectivity_block")
    assert classify_structure(result, set()) == "hold_inchikey_connectivity_disagreement"


def test_parse_failure_hold():
    result = qa(rdkit_status="rdkit_parse_failure", rdkit_canonical_isomeric_smiles="")
    assert classify_structure(result, {"CCO"}) == "hold_rdkit_parse_failure"


def test_multifragment_hold():
    result = qa(rdkit_fragment_count="2.0")
    assert classify_structure(result, set()) == "hold_multifragment"


@pytest.mark.parametrize("count", ["", "nan", "0", "1.5", "-1"])
def test_invalid_fragment_count_fails_closed(count):
    result = qa(rdkit_fragment_count=count)
    with pytest.raises(ValueError):
        classify_structure(result, set())


def test_parsed_missing_canonical_fails_closed():
    result = qa(rdkit_canonical_isomeric_smiles="")
    with pytest.raises(ValueError):
        classify_structure(result, set())


def test_unassigned_stereocenter_flagged():
    assert unspecified_stereogenic_elements("CC(O)N") >= 1


def test_assigned_stereocenter_not_flagged():
    assert unspecified_stereogenic_elements("C[C@H](O)N") == 0


def test_unassigned_double_bond_flagged():
    assert unspecified_stereogenic_elements("FC=CF") >= 1


def test_absent_structure_stereo_flag_unknown():
    assert unspecified_stereogenic_elements("") is None


def test_invalid_canonical_stereo_flag_rejected():
    with pytest.raises(ValueError):
        unspecified_stereogenic_elements("[C@@@H]")


def _csv_write(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as dest:
        writer = csv.DictWriter(dest, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_end_to_end_preserves_rows_and_labels(tmp_path):
    """Verify identity assignment does not change target, IC50 or row count."""
    from importlib.util import module_from_spec, spec_from_file_location
    from pathlib import Path

    script = Path(__file__).resolve().parents[2] / "scripts" / "apply_molecular_standardization.py"
    script = script.resolve()
    spec = spec_from_file_location("m18c_script", script)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    policy = tmp_path / "policy.csv"
    identity = tmp_path / "qa.csv"
    priority = tmp_path / "priority.csv"
    output = tmp_path / "output.csv"
    summary = tmp_path / "summary.json"
    collision = tmp_path / "collision.csv"
    base = {
        "smiles": "CCO", "inchi_key": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        "target": "EGFR", "m17_partition": "primary_exact_candidate",
        "ic50_raw": "100", "pic50": "7.0", "provenance_key": "pmid:test",
        "construct_category": "plain_name",
    }
    _csv_write(policy, ["reactant_set_id", *base], [
        {"reactant_set_id": "a", **base},
        {"reactant_set_id": "b", **base, "target": "HER2", "ic50_raw": "500", "pic50": "6.3"},
    ])
    ident = {
        **base, "rdkit_status": "parsed", "rdkit_canonical_isomeric_smiles": "CCO",
        "rdkit_inchi_key": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        "rdkit_fragment_count": "1.0", "source_key_comparison": "exact_match",
    }
    fields = ["reactant_set_id", "smiles", "inchi_key", "target", "m17_partition", "rdkit_status", "rdkit_canonical_isomeric_smiles", "rdkit_inchi_key", "rdkit_fragment_count", "source_key_comparison"]
    _csv_write(identity, fields, [
        {key: ({**ident, "reactant_set_id": "a"})[key] for key in fields},
        {key: ({**ident, "reactant_set_id": "b", "target": "HER2"})[key] for key in fields},
    ])
    _csv_write(priority, ["smiles", "identity_review_class", "review_priority"], [])
    result = module.run(policy, identity, priority, output, summary, collision, None)
    assert result["observations_retained"] == 2
    assert result["provisional_primary_exact_id_overlap_egfr_her2"] == 1
    with output.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert [row["ic50_raw"] for row in rows] == ["100", "500"]
    assert rows[0]["m18c_structure_id"] == rows[1]["m18c_structure_id"]
    assert json.loads(summary.read_text(encoding="utf-8"))["observations_retained"] == 2


def test_stale_priority_list_fails_closed(tmp_path):
    """The pipeline must reject a missing high-priority SMILES list."""
    from importlib.util import module_from_spec, spec_from_file_location
    from pathlib import Path

    script = (Path(__file__).resolve().parents[2] / "scripts" / "apply_molecular_standardization.py").resolve()
    spec = spec_from_file_location("m18c_script2", script)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    policy = tmp_path / "policy.csv"
    ident = tmp_path / "identity.csv"
    priority = tmp_path / "priority.csv"
    _csv_write(policy, ["reactant_set_id", "smiles", "inchi_key", "target", "m17_partition", "ic50_raw", "pic50", "provenance_key", "construct_category"], [])
    _csv_write(ident, ["reactant_set_id", "smiles", "inchi_key", "target", "m17_partition", "rdkit_status", "rdkit_canonical_isomeric_smiles", "rdkit_inchi_key", "rdkit_fragment_count", "source_key_comparison"], [
        {"reactant_set_id": "X", "smiles": "CCO", "inchi_key": "different", "target": "EGFR", "m17_partition": "primary_exact_candidate", "rdkit_status": "parsed", "rdkit_canonical_isomeric_smiles": "CCO", "rdkit_inchi_key": "something", "rdkit_fragment_count": "1", "source_key_comparison": "different_connectivity_block"}
    ])
    _csv_write(priority, ["smiles", "identity_review_class", "review_priority"], [])
    with pytest.raises(ValueError, match="High-priority SMILES"):
        module.run(policy, ident, priority, tmp_path / "out.csv", tmp_path / "sum.json", tmp_path / "collision.csv", None)
