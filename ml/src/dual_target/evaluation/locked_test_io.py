"""Explicitly gated access to frozen M2.1 test labels (M3.7d ONLY).

The ordinary load_modeling_pipeline() must remain test-blind. This separate
module is imported only in the one-time test execution branch, never preflight.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from ..data.modeling_dataset import (
    MoleculeRecord,
    _parse_label,
    _parse_mask,
    _read_manifest,
    sha256_file,
)

EXPECTED_DATASET = "83a64d02e45ccd9a6310947efcd5cdef9bddc9603edcd55e1d207eb1cafc18b8"
EXPECTED_MANIFEST = "05968ca1ed2a16ca8165e8a28ececa0f361fc9e61a64ac2e69abb1e0e4b7423a"
UNLOCK_TOKEN = "UNLOCK_M3_7D_TEST_ONCE"


def require_unlock(token: str) -> None:
    """Refuse to read any CSV data if explicit authorization is missing."""
    if token != UNLOCK_TOKEN:
        raise PermissionError("Frozen test labels require explicit M3.7d unlock authorization")


def load_locked_test_records(
    dataset_path: Path,
    manifest_path: Path,
    summary_path: Path,
    *,
    authorization_token: str,
    expected_dataset_sha256: str = EXPECTED_DATASET,
    expected_manifest_sha256: str = EXPECTED_MANIFEST,
    expected_test_n: int = 1087,
) -> tuple[MoleculeRecord, ...]:
    """Read exactly the frozen test rows, with their accepted labels, once unlocked.

    Uses the same mask/label parsing as M2.1. Train and validation labels are
    never parsed or returned by this specialized loader.
    """
    require_unlock(authorization_token)  # Must be first, before reading files.
    dataset_path, manifest_path, summary_path = map(
        Path, (dataset_path, manifest_path, summary_path)
    )
    if (sha256_file(dataset_path) != expected_dataset_sha256
            or sha256_file(manifest_path) != expected_manifest_sha256):
        raise ValueError("Frozen dataset/manifest SHA-256 mismatch")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    outputs = summary["outputs"]
    if (outputs["dataset_sha256"] != expected_dataset_sha256
            or outputs["manifest_sha256"] != expected_manifest_sha256):
        raise ValueError("Dataset/manifest differs from frozen M2.1 audit")
    manifest = _read_manifest(manifest_path)
    seen: set[str] = set()
    test: list[MoleculeRecord] = []
    counts = {"train": 0, "validation": 0, "test": 0}
    test_scaffolds: set[str] = set()
    other_scaffolds: set[str] = set()
    with dataset_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "m18c_structure_id", "canonical_isomeric_smiles", "training_split",
            "scaffold_split", "scaffold_group_id", "egfr_mask", "her2_mask",
            "egfr_pic50", "her2_pic50",
        }
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Frozen dataset missing expected M2.1 columns")
        for row in reader:
            sid = row["m18c_structure_id"]
            if sid in seen or sid not in manifest:
                raise ValueError(f"Duplicate or unrecognized M2.1 molecule: {sid}")
            seen.add(sid)
            frozen = manifest[sid]
            partition = frozen["training_split"]
            if (row["training_split"] != partition
                    or row["scaffold_split"] != partition
                    or row["scaffold_group_id"] != frozen["scaffold_group_id"]):
                raise ValueError(f"Frozen partition/scaffold mismatch: {sid}")
            masks = {}
            for target in ("egfr", "her2"):
                mask = _parse_mask(row[f"{target}_mask"])
                if str(mask) != frozen[f"{target}_mask"]:
                    raise ValueError(f"Frozen {target} mask mismatch: {sid}")
                masks[target] = mask
            counts[partition] += 1
            if partition != "test":
                other_scaffolds.add(row["scaffold_group_id"])
                continue  # Never parse train/validation labels in this reader.
            if not row["canonical_isomeric_smiles"]:
                raise ValueError("Test structure has no canonical SMILES")
            if masks["egfr"] + masks["her2"] < 1:
                raise ValueError("Test molecule has no eligible target")
            test_scaffolds.add(row["scaffold_group_id"])
            test.append(MoleculeRecord(
                structure_id=sid,
                smiles=row["canonical_isomeric_smiles"],
                scaffold_group_id=row["scaffold_group_id"],
                egfr_mask=masks["egfr"], her2_mask=masks["her2"],
                egfr_pic50=_parse_label(row["egfr_pic50"], masks["egfr"]),
                her2_pic50=_parse_label(row["her2_pic50"], masks["her2"]),
            ))
    if seen != set(manifest) or test_scaffolds & other_scaffolds:
        raise ValueError("Frozen test identity/scaffold isolation failed")
    audit = summary["audit"]
    if len(seen) != audit["total_unique_molecules"]:
        raise ValueError("Frozen M2.1 dataset molecule count mismatch")
    for partition, count in counts.items():
        if count != audit["by_partition"][partition]["molecules"]:
            raise ValueError(f"Frozen {partition} molecule count mismatch")
    if len(test) != expected_test_n or len({r.structure_id for r in test}) != len(test):
        raise ValueError("Expected exactly 1,087 distinct locked-test molecules")
    return tuple(test)
