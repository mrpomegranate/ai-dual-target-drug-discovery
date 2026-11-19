"""M3.7d test-label unlock and frozen M2.1 identity checks."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
from dual_target.evaluation.locked_test_io import (
    UNLOCK_TOKEN,
    load_locked_test_records,
    require_unlock,
)


def _csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path):
    manifest_rows = [
        {"m18c_structure_id": "a", "training_split": "train", "scaffold_group_id": "sa",
         "egfr_mask": "1", "her2_mask": "0"},
        {"m18c_structure_id": "b", "training_split": "validation", "scaffold_group_id": "sb",
         "egfr_mask": "0", "her2_mask": "1"},
        {"m18c_structure_id": "c", "training_split": "test", "scaffold_group_id": "sc",
         "egfr_mask": "1", "her2_mask": "1"},
        {"m18c_structure_id": "d", "training_split": "test", "scaffold_group_id": "sd",
         "egfr_mask": "1", "her2_mask": "0"},
    ]
    dataset_rows = []
    for row, labels in zip(manifest_rows, (("7", ""), ("", "6"), ("8", "9"), ("5", "")), strict=True):
        dataset_rows.append({
            **row,
            "canonical_isomeric_smiles": "CCO",
            "scaffold_split": row["training_split"],
            "egfr_pic50": labels[0], "her2_pic50": labels[1],
        })
    dataset = tmp_path / "data.csv"
    manifest = tmp_path / "manifest.csv"
    summary = tmp_path / "summary.json"
    _csv(dataset, dataset_rows)
    _csv(manifest, manifest_rows)
    summary.write_text(json.dumps({
        "outputs": {"dataset_sha256": _sha(dataset), "manifest_sha256": _sha(manifest)},
        "audit": {"total_unique_molecules": 4,
                  "by_partition": {p: {"molecules": n} for p, n in
                                   (("train", 1), ("validation", 1), ("test", 2))}},
    }))
    return dataset, manifest, summary, dataset_rows


def test_requires_token_before_touching_files(tmp_path):
    with pytest.raises(PermissionError, match="unlock"):
        load_locked_test_records(tmp_path / "missing.csv", tmp_path / "x.csv",
                                 tmp_path / "y.json", authorization_token="bad")
    with pytest.raises(PermissionError):
        require_unlock("")


def test_loads_test_only_with_masks(tmp_path):
    dataset, manifest, summary, _ = _fixture(tmp_path)
    result = load_locked_test_records(
        dataset, manifest, summary, authorization_token=UNLOCK_TOKEN,
        expected_dataset_sha256=_sha(dataset), expected_manifest_sha256=_sha(manifest),
        expected_test_n=2,
    )
    assert [r.structure_id for r in result] == ["c", "d"]
    assert result[0].her2_pic50 == 9.0
    assert result[1].her2_pic50 is None
    assert result[1].egfr_pic50 == 5.0


def test_mutated_dataset_fails_hash(tmp_path):
    dataset, manifest, summary, _ = _fixture(tmp_path)
    before = _sha(dataset)
    dataset.write_text(dataset.read_text() + "\n")
    with pytest.raises(ValueError, match="SHA-256"):
        load_locked_test_records(dataset, manifest, summary,
                                 authorization_token=UNLOCK_TOKEN,
                                 expected_dataset_sha256=before,
                                 expected_manifest_sha256=_sha(manifest),
                                 expected_test_n=2)


def test_changed_test_mask_fails_manifest(tmp_path):
    dataset, manifest, summary, rows = _fixture(tmp_path)
    rows[2]["her2_mask"] = "0"
    _csv(dataset, rows)
    summary.write_text(json.dumps({
        "outputs": {"dataset_sha256": _sha(dataset), "manifest_sha256": _sha(manifest)},
        "audit": {"total_unique_molecules": 4,
                  "by_partition": {p: {"molecules": n} for p, n in
                                   (("train", 1), ("validation", 1), ("test", 2))}},
    }))
    with pytest.raises(ValueError, match="mask mismatch"):
        load_locked_test_records(dataset, manifest, summary,
                                 authorization_token=UNLOCK_TOKEN,
                                 expected_dataset_sha256=_sha(dataset),
                                 expected_manifest_sha256=_sha(manifest),
                                 expected_test_n=2)


def test_test_count_fails_closed(tmp_path):
    dataset, manifest, summary, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="1,087"):
        load_locked_test_records(dataset, manifest, summary,
                                 authorization_token=UNLOCK_TOKEN,
                                 expected_dataset_sha256=_sha(dataset),
                                 expected_manifest_sha256=_sha(manifest),
                                 expected_test_n=1087)


def test_missing_test_label_fails_closed(tmp_path):
    dataset, manifest, summary, rows = _fixture(tmp_path)
    rows[2]["her2_pic50"] = ""
    _csv(dataset, rows)
    summary.write_text(json.dumps({
        "outputs": {"dataset_sha256": _sha(dataset), "manifest_sha256": _sha(manifest)},
        "audit": {"total_unique_molecules": 4,
                  "by_partition": {p: {"molecules": n} for p, n in
                                   (("train", 1), ("validation", 1), ("test", 2))}},
    }))
    with pytest.raises(ValueError, match="missing pIC50"):
        load_locked_test_records(dataset, manifest, summary,
                                 authorization_token=UNLOCK_TOKEN,
                                 expected_dataset_sha256=_sha(dataset),
                                 expected_manifest_sha256=_sha(manifest),
                                 expected_test_n=2)
