
"""Tests for the frozen M2.1 modeling dataset loader."""

import csv
import json

import pytest
from dual_target.data import modeling_dataset as md

DATA_COLUMNS = [
    "m18c_structure_id",
    "canonical_isomeric_smiles",
    "training_split",
    "scaffold_split",
    "scaffold_group_id",
    "egfr_mask",
    "her2_mask",
    "egfr_pic50",
    "her2_pic50",
]

MANIFEST_COLUMNS = [
    "m18c_structure_id",
    "training_split",
    "scaffold_group_id",
    "egfr_mask",
    "her2_mask",
]


def write_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def seal(case, monkeypatch):
    """Recalculate fixture hashes after intentional test modifications."""
    write_csv(case["dataset"], case["rows"], DATA_COLUMNS)
    write_csv(case["manifest"], case["manifest_rows"], MANIFEST_COLUMNS)

    manifest_hash = md.sha256_file(case["manifest"])
    dataset_hash = md.sha256_file(case["dataset"])

    case["summary_data"]["outputs"] = {
        "manifest_sha256": manifest_hash,
        "dataset_sha256": dataset_hash,
    }

    case["summary"].write_text(
        json.dumps(case["summary_data"]), encoding="utf-8"
    )
    monkeypatch.setattr(md, "FROZEN_MANIFEST_SHA256", manifest_hash)


def load(case):
    return md.load_modeling_data(
        case["dataset"], case["manifest"], case["summary"]
    )


@pytest.fixture
def case(tmp_path, monkeypatch):
    rows = [
        {
            "m18c_structure_id": sid,
            "canonical_isomeric_smiles": smiles,
            "training_split": split,
            "scaffold_split": split,
            "scaffold_group_id": scaffold,
            "egfr_mask": em,
            "her2_mask": hm,
            "egfr_pic50": ep,
            "her2_pic50": hp,
        }
        for sid, smiles, split, scaffold, em, hm, ep, hp in [
            ("a", "CCO", "train", "g1", "1", "0", "7.0", ""),
            ("b", "CCN", "train", "g2", "0", "1", "", "6.5"),
            ("c", "c1ccccc1", "validation", "g3", "1", "1", "8.0", "7.0"),
            ("d", "CCC", "test", "g4", "1", "0", "TEST_HIDDEN", ""),
        ]
    ]

    result = {
        "rows": rows,
        "manifest_rows": [
            {key: row[key] for key in MANIFEST_COLUMNS}
            for row in rows
        ],
        "dataset": tmp_path / "dataset.csv",
        "manifest": tmp_path / "manifest.csv",
        "summary": tmp_path / "summary.json",
        "summary_data": {
            "audit": {
                "total_unique_molecules": 4,
                "by_partition": {
                    "train": {"molecules": 2},
                    "validation": {"molecules": 1},
                    "test": {"molecules": 1},
                },
            }
        },
    }

    seal(result, monkeypatch)
    return result


def test_train_validation_only(case):
    data = load(case)

    assert len(data.train) == 2
    assert len(data.validation) == 1
    assert {r.structure_id for r in data.train} == {"a", "b"}
    assert data.validation[0].structure_id == "c"

    # Test molecule d must not be returned.
    assert "d" not in {
        r.structure_id for r in data.train + data.validation
    }


def test_masks_and_labels_preserved(case):
    data = load(case)

    assert data.train[0].egfr_pic50 == 7.0
    assert data.train[0].her2_pic50 is None
    assert data.train[1].egfr_pic50 is None
    assert data.train[1].her2_pic50 == 6.5


def test_test_labels_never_parsed(case):
    # The test label is deliberately nonnumeric.
    # Loading succeeds because test potency is never parsed.
    data = load(case)
    assert len(data.train) + len(data.validation) == 3


def test_frozen_manifest_hash_enforced(case, monkeypatch):
    monkeypatch.setattr(md, "FROZEN_MANIFEST_SHA256", "0" * 64)

    with pytest.raises(ValueError, match="checksum mismatch"):
        load(case)


def test_dataset_modification_detected(case):
    case["rows"][0]["egfr_pic50"] = "9.0"
    write_csv(case["dataset"], case["rows"], DATA_COLUMNS)

    with pytest.raises(ValueError, match="Dataset does not match"):
        load(case)


def test_scaffold_leakage_rejected(case, monkeypatch):
    # Place validation molecule in a training scaffold group.
    case["manifest_rows"][2]["scaffold_group_id"] = "g1"
    seal(case, monkeypatch)

    with pytest.raises(ValueError, match="Scaffold leakage"):
        load(case)


def test_changed_target_mask_rejected(case, monkeypatch):
    case["rows"][0]["egfr_mask"] = "0"
    seal(case, monkeypatch)

    with pytest.raises(ValueError, match="Target mask changed"):
        load(case)


def test_missing_measured_label_rejected(case, monkeypatch):
    case["rows"][0]["egfr_pic50"] = ""
    seal(case, monkeypatch)

    with pytest.raises(ValueError, match="missing pIC50"):
        load(case)


def test_masked_target_cannot_have_label(case, monkeypatch):
    case["rows"][1]["egfr_pic50"] = "5.5"
    seal(case, monkeypatch)

    with pytest.raises(ValueError, match="Masked target"):
        load(case)


def test_nonfinite_potency_rejected(case, monkeypatch):
    case["rows"][0]["egfr_pic50"] = "nan"
    seal(case, monkeypatch)

    with pytest.raises(ValueError, match="must be finite"):
        load(case)


def test_changed_partition_rejected(case, monkeypatch):
    case["rows"][2]["training_split"] = "train"
    seal(case, monkeypatch)

    with pytest.raises(ValueError, match="Split assignment mismatch"):
        load(case)


def test_duplicate_molecule_rejected(case, monkeypatch):
    case["rows"].append(case["rows"][0].copy())
    seal(case, monkeypatch)

    with pytest.raises(ValueError, match="Duplicate or unknown"):
        load(case)
