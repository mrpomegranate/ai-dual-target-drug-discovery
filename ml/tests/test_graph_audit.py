"""Synthetic-only tests of fit-role molecular graph audit checks."""

from __future__ import annotations

import runpy
from pathlib import Path

from dual_target.data.modeling_dataset import MoleculeRecord

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "audit_molecular_graphs.py"
audit_fit_graphs = runpy.run_path(str(_SCRIPT))["audit_fit_graphs"]


def _record(
    sid: str, smiles: str, egfr: float | None, her2: float | None
) -> MoleculeRecord:
    return MoleculeRecord(
        structure_id=sid,
        smiles=smiles,
        scaffold_group_id=f"scaffold_{sid}",
        egfr_mask=int(egfr is not None),
        her2_mask=int(her2 is not None),
        egfr_pic50=egfr,
        her2_pic50=her2,
    )


def test_audit_aggregate_counts_and_mask_preservation() -> None:
    rows = (
        _record("A", "CCO", 7.0, None),
        _record("B", "C1CCCCC1", None, 6.0),
        _record("C", "[Na+]", 7.5, 6.5),
    )
    report = audit_fit_graphs(
        rows,
        expected_counts={
            "molecules": 3,
            "egfr_labels": 2,
            "her2_labels": 2,
            "paired_labels": 1,
        },
    )
    assert report["passed"]
    assert report["graphs_converted"] == 3
    assert report["graphs_failed"] == 0
    assert report["atom_count_distribution"]["max"] == 6
    assert report["single_atom_graphs"] == 1
    assert report["edgeless_graphs"] == 1
    assert report["total_atoms"] == 10
    assert report["total_directed_edges"] == 16
    assert report["label_counts"]["dual_active"] == 1


def test_invalid_smiles_records_failure_and_does_not_succeed() -> None:
    rows = (_record("bad", "not_a_smiles", 7.0, None),)
    report = audit_fit_graphs(rows)
    assert not report["passed"]
    assert report["graphs_converted"] == 0
    assert report["graphs_failed"] == 1
    assert report["failure_examples_first_25"][0]["structure_id"] == "bad"


def test_mismatched_expected_counts_fail_audit() -> None:
    rows = (_record("A", "CCO", 7.0, None),)
    report = audit_fit_graphs(rows, expected_counts={"molecules": 5364})
    assert not report["passed"]
    assert report["graphs_converted"] == 1
    assert report["expected_count_mismatches"]
