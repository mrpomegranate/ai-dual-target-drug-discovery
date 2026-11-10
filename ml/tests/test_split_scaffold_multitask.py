"""Synthetic-only M2.1 split integrity tests; no BindingDB records embedded."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from rdkit import rdBase

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "split_scaffold_multitask.py"
SPEC = importlib.util.spec_from_file_location("split_scaffold_multitask", SCRIPT)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def sid(smiles: str) -> str:
    return "RDKISO1-" + hashlib.sha256(
        b"rdkit-isomeric-smiles-policy-v1\0" + smiles.encode("utf-8")
    ).hexdigest()


def row(smiles: str, egfr: str = "7.2", her2: str = "6.8") -> dict[str, str]:
    return {
        "m18c_structure_id": sid(smiles), "canonical_isomeric_smiles": smiles,
        "training_split": "NOT_ASSIGNED", "scaffold_split": "NOT_ASSIGNED",
        "egfr_mask": "1" if egfr else "0", "egfr_pic50": egfr,
        "her2_mask": "1" if her2 else "0", "her2_pic50": her2,
        "n_available_regression_tasks": str(bool(egfr) + bool(her2)),
        "paired_activity_class": "dual_active" if egfr and her2 else "not_evaluable_missing_task_label",
    }


def samples() -> list[dict[str, str]]:
    return [
        row("c1ccccc1O"), row("Cc1ccccc1"),  # shared benzene Murcko core
        row("C1CCCCC1"), row("c1ccncc1"), row("c1ccoc1"), row("C1CC1"),
        row("c1ccc2ccccc2c1"), row("C1CCC1"), row("CCO"), row("CCN"),
        row("CCCO"), row("CCCN"),
    ]


def csv_file(path: Path, records: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def test_same_scaffold_never_crosses_partitions():
    groups, identity = mod.prepare(samples())
    assignments = mod.assign_groups(groups, 714)
    assert identity[sid("c1ccccc1O")][0] == identity[sid("Cc1ccccc1")][0]
    assert assignments[identity[sid("c1ccccc1O")][0]] == assignments[identity[sid("Cc1ccccc1")][0]]
    report = mod.audit(samples(), identity, groups, assignments)
    assert report["zero_molecule_id_overlap"]
    assert report["zero_scaffold_group_id_overlap"]
    assert sum(x["molecules"] for x in report["by_partition"].values()) == len(samples())


def test_acyclic_empty_scaffolds_get_per_molecule_groups():
    g1, s1, kind1 = mod.compute_scaffold("CCO", sid("CCO"))
    g2, s2, kind2 = mod.compute_scaffold("CCN", sid("CCN"))
    assert g1 != g2
    assert s1 == s2 == ""
    assert kind1 == kind2 == "acyclic_identity_group"


def test_stereo_ignored_for_scaffold_group():
    a, _, _ = mod.compute_scaffold("C[C@H]1CCCC1", sid("C[C@H]1CCCC1"))
    b, _, _ = mod.compute_scaffold("C[C@@H]1CCCC1", sid("C[C@@H]1CCCC1"))
    assert a == b


def test_seed_is_deterministic():
    groups, _ = mod.prepare(samples())
    assert mod.assign_groups(groups, 714) == mod.assign_groups(groups, 714)


def test_potency_does_not_influence_splitting():
    old = samples()
    changed = samples()
    changed[0]["egfr_pic50"] = "8.4"  # remains active
    changed[0]["her2_pic50"] = "8.0"  # remains active
    groups1, _ = mod.prepare(old)
    groups2, _ = mod.prepare(changed)
    assert mod.assign_groups(groups1, 714) == mod.assign_groups(groups2, 714)


def test_missing_target_stays_blank():
    record = row("CCO", her2="")
    assert mod.validate_row(record, set(), set()) == (1, 0)
    record["her2_pic50"] = "0"
    with pytest.raises(ValueError, match="blank pIC50"):
        mod.validate_row(record, set(), set())


def test_malformed_identity_rejected():
    record = row("CCO")
    record["m18c_structure_id"] = sid("CCN")
    with pytest.raises(ValueError, match="identity digest mismatch"):
        mod.prepare([record])


def test_existing_split_rejected():
    record = row("CCO")
    record["training_split"] = "train"
    with pytest.raises(ValueError, match="already split"):
        mod.validate_row(record, set(), set())


def test_duplicate_molecule_rejected():
    a = row("CCO")
    with pytest.raises(ValueError, match="Duplicate molecular identity"):
        mod.prepare([a, dict(a)])


def test_threshold_boundary_and_class_validated():
    r = row("CCO", egfr="6.0", her2="7.0")
    r["paired_activity_class"] = "HER2_only_active"
    mod.validate_row(r, set(), set())
    r["paired_activity_class"] = "dual_active"
    with pytest.raises(ValueError, match="Paired classification inconsistent"):
        mod.validate_row(r, set(), set())


def test_pipeline_repeatable_and_preserves_labels(tmp_path: Path):
    src = tmp_path / "m11.csv"
    csv_file(src, samples())
    a = tmp_path / "a"
    b = tmp_path / "b"
    x = mod.run(src, None, a / "dataset.csv", a / "manifest.csv", a / "summary.json", 714)
    y = mod.run(src, None, b / "dataset.csv", b / "manifest.csv", b / "summary.json", 714)
    assert x["outputs"]["manifest_sha256"] == y["outputs"]["manifest_sha256"]
    assert x["outputs"]["dataset_sha256"] == y["outputs"]["dataset_sha256"]
    with (a / "dataset.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert sorted((r["m18c_structure_id"], r["egfr_pic50"], r["her2_pic50"]) for r in rows) == sorted(
        (r["m18c_structure_id"], r["egfr_pic50"], r["her2_pic50"]) for r in samples()
    )
    assert {r["training_split"] for r in rows} == {"train", "validation", "test"}
    assert all(r["scaffold_split"] == r["training_split"] for r in rows)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        mod.run(src, None, a / "dataset.csv", a / "manifest.csv", a / "summary.json", 714)


def test_m11_summary_counts_checked(tmp_path: Path):
    rows = samples()
    path = tmp_path / "m11_summary.json"
    source = {
        "analysis": {"unique_accepted_structures": len(rows), "EGFR_labels": len(rows),
                     "HER2_labels": len(rows), "dual_labeled_structures": len(rows)},
        "inputs": {"rdkit_version_provenance": rdBase.rdkitVersion},
    }
    path.write_text(json.dumps(source), encoding="utf-8")
    assert mod.check_m11_summary(path, rows)["m11_source_rdkit_version"] == rdBase.rdkitVersion
    source["analysis"]["EGFR_labels"] -= 1
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="M1.11 summary mismatch"):
        mod.check_m11_summary(path, rows)
