
"""Load the frozen M2.1 train/validation data without exposing test labels."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

FROZEN_MANIFEST_SHA256 = (
    "05968ca1ed2a16ca8165e8a28ececa0f361fc9e61a64ac2e69abb1e0e4b7423a"
)

PARTITIONS = {"train", "validation", "test"}
TARGETS = ("egfr", "her2")


@dataclass(frozen=True)
class MoleculeRecord:
    structure_id: str
    smiles: str
    scaffold_group_id: str
    egfr_mask: int
    her2_mask: int
    egfr_pic50: float | None
    her2_pic50: float | None


@dataclass(frozen=True)
class ModelingData:
    train: tuple[MoleculeRecord, ...]
    validation: tuple[MoleculeRecord, ...]
    manifest_sha256: str
    dataset_sha256: str


def sha256_file(path: Path) -> str:
    """Compute a file's SHA-256 checksum."""
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _read_manifest(path: Path) -> dict[str, dict[str, str]]:
    """Validate molecule-to-scaffold-to-partition assignments."""
    rows = {}
    scaffold_partitions = {}

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "m18c_structure_id",
            "training_split",
            "scaffold_group_id",
            "egfr_mask",
            "her2_mask",
        }
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Split manifest has missing columns.")

        for row in reader:
            sid = row["m18c_structure_id"]
            split = row["training_split"]
            scaffold = row["scaffold_group_id"]

            if not sid or not scaffold or split not in PARTITIONS:
                raise ValueError("Invalid split manifest entry.")

            if sid in rows:
                raise ValueError(f"Duplicate molecule ID: {sid}")

            previous = scaffold_partitions.setdefault(scaffold, split)
            if previous != split:
                raise ValueError(f"Scaffold leakage detected: {scaffold}")

            rows[sid] = row

    if not rows or {r["training_split"] for r in rows.values()} != PARTITIONS:
        raise ValueError("Manifest must contain all three partitions.")

    return rows


def _parse_mask(value: str) -> int:
    if value not in {"0", "1"}:
        raise ValueError(f"Invalid target mask: {value!r}")
    return int(value)


def _parse_label(value: str, mask: int) -> float | None:
    """Never interpret an absent target label as zero."""
    if mask == 0:
        if value.strip():
            raise ValueError("Masked target unexpectedly contains a label.")
        return None

    if not value.strip():
        raise ValueError("Measured target has a missing pIC50.")

    result = float(value)
    if not math.isfinite(result):
        raise ValueError("pIC50 must be finite.")

    return result


def load_modeling_data(
    dataset_path: str | Path,
    manifest_path: str | Path,
    summary_path: str | Path,
) -> ModelingData:
    """Read train and validation only; test potency labels stay reserved."""
    dataset_path = Path(dataset_path)
    manifest_path = Path(manifest_path)
    summary_path = Path(summary_path)

    manifest_hash = sha256_file(manifest_path)
    dataset_hash = sha256_file(dataset_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    if manifest_hash != FROZEN_MANIFEST_SHA256:
        raise ValueError("Frozen M2.1 manifest checksum mismatch.")

    outputs = summary["outputs"]
    if outputs["manifest_sha256"] != manifest_hash:
        raise ValueError("Manifest does not match M2.1 summary.")
    if outputs["dataset_sha256"] != dataset_hash:
        raise ValueError("Dataset does not match M2.1 summary.")

    manifest = _read_manifest(manifest_path)
    partitions: dict[str, list[MoleculeRecord]] = {
        "train": [],
        "validation": [],
    }
    seen_ids = set()
    counts = {part: 0 for part in PARTITIONS}

    with dataset_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "m18c_structure_id",
            "canonical_isomeric_smiles",
            "training_split",
            "scaffold_split",
            "scaffold_group_id",
            "egfr_mask",
            "her2_mask",
            "egfr_pic50",
            "her2_pic50",
        }

        if not required.issubset(reader.fieldnames or []):
            raise ValueError("M2.1 dataset has missing columns.")

        for row in reader:
            sid = row["m18c_structure_id"]

            if sid in seen_ids or sid not in manifest:
                raise ValueError(f"Duplicate or unknown molecule: {sid}")
            seen_ids.add(sid)

            entry = manifest[sid]
            split = entry["training_split"]

            if (
                row["training_split"] != split
                or row["scaffold_split"] != split
                or row["scaffold_group_id"] != entry["scaffold_group_id"]
            ):
                raise ValueError(f"Split assignment mismatch: {sid}")

            masks = {}
            for target in TARGETS:
                mask = _parse_mask(row[f"{target}_mask"])
                if str(mask) != entry[f"{target}_mask"]:
                    raise ValueError(f"Target mask changed: {sid}")
                masks[target] = mask

            counts[split] += 1

            # Do not parse, store, or return test potency labels.
            if split == "test":
                continue

            record = MoleculeRecord(
                structure_id=sid,
                smiles=row["canonical_isomeric_smiles"],
                scaffold_group_id=row["scaffold_group_id"],
                egfr_mask=masks["egfr"],
                her2_mask=masks["her2"],
                egfr_pic50=_parse_label(
                    row["egfr_pic50"], masks["egfr"]
                ),
                her2_pic50=_parse_label(
                    row["her2_pic50"], masks["her2"]
                ),
            )
            partitions[split].append(record)

    if seen_ids != set(manifest):
        raise ValueError("Dataset and manifest molecule IDs differ.")

    audit = summary["audit"]
    if len(seen_ids) != audit["total_unique_molecules"]:
        raise ValueError("Total molecule count differs from frozen audit.")

    for split in PARTITIONS:
        if counts[split] != audit["by_partition"][split]["molecules"]:
            raise ValueError(f"Partition count mismatch: {split}")

    return ModelingData(
        train=tuple(partitions["train"]),
        validation=tuple(partitions["validation"]),
        manifest_sha256=manifest_hash,
        dataset_sha256=dataset_hash,
    )
