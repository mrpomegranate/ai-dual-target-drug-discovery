"""Synthetic-only safety tests: no private BindingDB records embedded in source."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "assemble_multitask_dataset.py"
SPEC = importlib.util.spec_from_file_location("assemble_multitask_dataset", SCRIPT)
mod = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(mod)


def sid(smiles: str) -> str:
    return "RDKISO1-" + hashlib.sha256(
        b"rdkit-isomeric-smiles-policy-v1\0" + smiles.encode("utf-8")
    ).hexdigest()


def exact(rid: str, target: str, smiles: str, pic50: float) -> dict[str, str]:
    return {
        "reactant_set_id": rid, "target": target,
        "m17_partition": "primary_exact_candidate", "m18c_status": "provisional_exact_structure",
        "m18c_structure_id": sid(smiles), "m18c_candidate_isomeric_smiles": smiles,
        "pic50": str(pic50), "target_name": "Plain target name",
        "construct_category": "plain_name", "mutation_signature": "",
        "measurement_status": "exact", "ic50_raw": "100", "ic50_nm": "100",
        "ic50_relation": "=", "source": "synthetic", "provenance_key": "pmid:fake",
    }


def candidate(rid: str, target: str, smiles: str, pic50: float,
              status: str = "provisional_single_exact", overlap: bool = False) -> dict[str, str]:
    accepted = status in mod.ACCEPT
    return {
        "target": target, "m18c_structure_id": sid(smiles),
        "canonical_isomeric_smiles": smiles, "n_exact_observations": "1",
        "n_provenance_keys": "1", "m10_policy_status": status,
        "m10_provisional_pic50": str(pic50) if accepted else "",
        "m10_provisional_activity": ("active" if pic50 > 6 else "inactive") if accepted else "",
        "m10_label_support": "unreplicated_unverified" if accepted else "held_no_label",
        "m10_target_construct_verification": "NOT_sequence_or_assay_verified",
        "m10_exact_observation_ids": rid,
        "has_exact_activity_conflict": "False", "has_censored_activity_conflict": "False",
        "has_annotated_construct_for_same_structure_target": str(overlap),
        "min_exact_pic50": str(pic50), "max_exact_pic50": str(pic50),
        "exact_pic50_range": "0.0",
    }


def reserve(rid: str, target: str, smiles: str) -> dict[str, str]:
    row = exact(rid, target, smiles, 7.0)
    row.update({"m17_partition": "secondary_construct_reserve",
                "target_name": "EGFR L858R", "construct_category": "annotated_construct"})
    return row


def synthetic():
    source = [exact("11", "EGFR", "CCO", 7.0), exact("12", "HER2", "CCO", 6.5),
              exact("13", "EGFR", "CCN", 5.0), exact("14", "HER2", "CCC", 7.2),
              exact("15", "HER2", "CCN", 8.0), reserve("99", "EGFR", "CCO")]
    candidates = [candidate("11", "EGFR", "CCO", 7.0, overlap=True),
                  candidate("12", "HER2", "CCO", 6.5),
                  candidate("13", "EGFR", "CCN", 5.0, status="hold_moderate_disagreement"),
                  candidate("14", "HER2", "CCC", 7.2),
                  candidate("15", "HER2", "CCN", 8.0)]
    index, reserved, _ = mod.source_index(source)
    return candidates, index, reserved


def assemble():
    return mod.build_dataset(*synthetic())


def test_identity_versioned_sha256():
    mod.check_identity(sid("CCO"), "CCO")


def test_identity_modified_smiles_fails():
    with pytest.raises(ValueError, match="digest disagrees"):
        mod.check_identity(sid("CCO"), "CCC")


def test_masked_rows_only_have_accepted_labels():
    assembled, _, _, _ = assemble()
    keyed = {r["m18c_structure_id"]: r for r in assembled}
    assert len(assembled) == 3
    assert keyed[sid("CCO")]["egfr_mask"] == keyed[sid("CCO")]["her2_mask"] == 1
    assert keyed[sid("CCN")]["egfr_mask"] == 0
    assert keyed[sid("CCN")]["egfr_pic50"] == ""
    assert keyed[sid("CCN")]["egfr_status"] == "hold_moderate_disagreement"
    assert keyed[sid("CCC")]["egfr_status"] == "no_primary_exact_evidence"
    assert keyed[sid("CCC")]["egfr_pic50"] == ""


def test_dual_activity_derived_only_for_two_accepted_tasks():
    assembled, _, _, qc = assemble()
    row = next(r for r in assembled if r["m18c_structure_id"] == sid("CCO"))
    assert row["paired_activity_class"] == "dual_active"
    assert qc["dual_labeled_structures"] == 1
    assert qc["HER2_only_labeled_structures"] == 2


def test_source_held_rows_preserved_in_pair_audit():
    _, decisions, _, _ = assemble()
    held = next(r for r in decisions if r["m10_policy_status"] == "hold_moderate_disagreement")
    assert held["m11_regression_mask"] == 0
    assert held["m11_selected_in_multitask_dataset"] == 1
    assert held["original_exact_observation_ids"] == "13"


def test_construct_reserve_not_in_primary_dataset():
    assembled, _, reserve_manifest, qc = assemble()
    assert len(reserve_manifest) == 1
    assert reserve_manifest[0]["eligible_for_primary_training"] == "False"
    assert qc["annotated_construct_overlap_accepted_pairs"] == 1
    assert len(assembled) == 3


def test_duplicate_candidate_pair_rejected():
    rows, index, reserved = synthetic()
    with pytest.raises(ValueError, match="Duplicate molecule-target"):
        mod.build_dataset(rows + [copy.deepcopy(rows[0])], index, reserved)


def test_duplicate_source_observation_rejected():
    row = exact("11", "EGFR", "CCO", 7.0)
    with pytest.raises(ValueError, match="duplicate reactant_set_id"):
        mod.source_index([row, copy.deepcopy(row)])


def test_wrong_source_target_pair_rejected():
    rows, index, reserved = synthetic()
    rows[0]["target"] = "HER2"
    with pytest.raises(ValueError, match="Original exact observation"):
        mod.build_dataset(rows, index, reserved)


def test_wrong_pic50_extrema_rejected():
    rows, index, reserved = synthetic()
    rows[0]["max_exact_pic50"] = "9.0"
    with pytest.raises(ValueError, match="maximum pIC50"):
        mod.build_dataset(rows, index, reserved)


def test_accepted_conflict_flag_rejected():
    rows, index, reserved = synthetic()
    rows[0]["has_censored_activity_conflict"] = "True"
    with pytest.raises(ValueError, match="unresolved activity conflict"):
        mod.build_dataset(rows, index, reserved)


def test_held_numeric_label_rejected():
    rows, index, reserved = synthetic()
    rows[2]["m10_provisional_pic50"] = "5.0"
    with pytest.raises(ValueError, match="Held pair must not"):
        mod.build_dataset(rows, index, reserved)


def test_unverified_construct_assumption_enforced():
    rows, index, reserved = synthetic()
    rows[0]["m10_target_construct_verification"] = "WT verified"
    with pytest.raises(ValueError, match="Unexpected assay verification"):
        mod.build_dataset(rows, index, reserved)


def test_missing_source_observation_rejected():
    rows, index, reserved = synthetic()
    index.pop("15")
    with pytest.raises(ValueError, match="not in expected"):
        mod.build_dataset(rows, index, reserved)


def test_repeated_candidate_original_id_rejected():
    rows, index, reserved = synthetic()
    rows[0]["m10_exact_observation_ids"] = "11;11"
    rows[0]["n_exact_observations"] = "2"
    with pytest.raises(ValueError, match="Invalid original observation"):
        mod.build_dataset(rows, index, reserved)


def test_hold_only_structure_stays_out_of_model_rows():
    rows, index, reserved = synthetic()
    rows[4]["m10_policy_status"] = "hold_moderate_disagreement"
    rows[4]["m10_provisional_pic50"] = ""
    rows[4]["m10_provisional_activity"] = ""
    assembled, decisions, _, qc = mod.build_dataset(rows, index, reserved)
    assert sid("CCN") not in {r["m18c_structure_id"] for r in assembled}
    assert qc["held_only_structure_ids"] == 1
    assert sum(x["m11_selected_in_multitask_dataset"] == 0 for x in decisions) == 2


def test_strict_activity_cutoff_at_pic50_six():
    rows, index, reserved = synthetic()
    rows[1]["m10_provisional_pic50"] = "6.0"
    rows[1]["m10_provisional_activity"] = "inactive"
    rows[1]["min_exact_pic50"] = "6.0"
    rows[1]["max_exact_pic50"] = "6.0"
    index["12"]["pic50"] = "6.0"
    assembled, _, _, _ = mod.build_dataset(rows, index, reserved)
    egfr_her2 = next(x for x in assembled if x["m18c_structure_id"] == sid("CCO"))
    assert egfr_her2["paired_activity_class"] == "EGFR_only_active"
