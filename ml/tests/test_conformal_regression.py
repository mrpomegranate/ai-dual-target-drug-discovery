"""Conformal regression boundary, finite-sample quantiles and coverage checks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest
from dual_target.calibration.conformal_regression import (
    conformal_quantile,
    fit_split_conformal,
    summarize_interval_coverage,
)


@dataclass(frozen=True)
class MoleculeRecord:
    structure_id: str
    smiles: str
    scaffold_group_id: str
    egfr_mask: int
    her2_mask: int
    egfr_pic50: float | None
    her2_pic50: float | None


def molecule(sid, scaffold, egfr, her2=None):
    return MoleculeRecord(
        structure_id=sid, smiles="CC", scaffold_group_id=scaffold,
        egfr_mask=int(egfr is not None), her2_mask=int(her2 is not None),
        egfr_pic50=egfr, her2_pic50=her2,
    )


def example_split():
    fit = (molecule("fit", "F", 7),)
    calibration = tuple(
        molecule(f"cal{i}", f"C{i // 2}", float(i + 5), float(i + 6) if i % 2 else None)
        for i in range(12)
    )
    threshold = (molecule("threshold", "T", 8),)
    assignments = tuple(SimpleNamespace(structure_id=r.structure_id,
              scaffold_group_id=r.scaffold_group_id, role=role)
        for role, records in (("fit", fit), ("calibration", calibration), ("threshold", threshold))
        for r in records)
    canonical = [[a.structure_id, a.scaffold_group_id, a.role]
                 for a in sorted(assignments, key=lambda a: a.structure_id)]
    assignment_hash = hashlib.sha256(
        json.dumps(canonical, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    split = SimpleNamespace(
        fit=fit, calibration=calibration, threshold=threshold,
        assignments=assignments, assignment_sha256=assignment_hash,
        dataset_sha256="dataset", manifest_sha256="manifest",
    )
    predictions = {r.structure_id: r.egfr_pic50 + i * .1 for i, r in enumerate(calibration)}
    return split, predictions


def test_conformal_quantile_uses_n_plus_one():
    assert conformal_quantile(np.arange(10), alpha=.2) == 8.


def test_conformal_quantile_rejects_unavailable_finite_interval():
    with pytest.raises(ValueError, match="unavailable"):
        conformal_quantile(np.ones(8), alpha=.1)


@pytest.mark.parametrize("residuals,alpha", [([-1, 2], .1), ([1, float("nan")], .2), ([], .2)])
def test_conformal_quantile_rejects_bad_residuals(residuals, alpha):
    with pytest.raises(ValueError):
        conformal_quantile(residuals, alpha=alpha)


def test_calibrator_only_uses_labelled_calibration_and_preserves_provenance():
    split, prediction = example_split()
    calibrated = fit_split_conformal(
        split, prediction, target="egfr", model_name="gine",
        expected_assignment_sha256=split.assignment_sha256,
    )
    assert calibrated.n_labels == 12
    assert calibrated.n_scaffolds == 6
    assert calibrated.absolute_error_quantile == pytest.approx(1.1)
    assert calibrated.dataset_sha256 == "dataset"
    assert calibrated.stage_c_assignment_sha256 == split.assignment_sha256
    assert np.allclose(calibrated.intervals([7]), [[5.9, 8.1]])


def test_calibrator_skips_missing_target_labels():
    split, predictions = example_split()
    calibrated = fit_split_conformal(
        split, predictions, target="her2", model_name="rf", alpha=.25,
        expected_assignment_sha256=split.assignment_sha256,
    )
    assert calibrated.n_labels == 6


def test_calibrator_rejects_calibration_identity_mismatch():
    split, predictions = example_split()
    with pytest.raises(ValueError, match="EXACTLY"):
        fit_split_conformal(split, {**predictions, "fit": 7.}, target="egfr",
                            model_name="rf", expected_assignment_sha256=split.assignment_sha256)


def test_calibrator_rejects_stage_c_hash_mismatch():
    split, predictions = example_split()
    with pytest.raises(ValueError, match="digest"):
        fit_split_conformal(split, predictions, target="egfr", model_name="rf")


def test_calibrator_rejects_scaffold_leakage():
    split, predictions = example_split()
    split.calibration = (replace(split.calibration[0], scaffold_group_id="F"),) + split.calibration[1:]
    with pytest.raises(ValueError, match="Scaffold leakage"):
        fit_split_conformal(split, predictions, target="egfr", model_name="rf",
                            expected_assignment_sha256=split.assignment_sha256)


def test_independent_coverage_summary_and_scaffold_min_n_flag():
    split, predictions = example_split()
    calibrator = fit_split_conformal(split, predictions, target="egfr", model_name="rf",
                                     expected_assignment_sha256=split.assignment_sha256)
    rows = (molecule("dev1", "V", 7), molecule("dev2", "V", 9),
            molecule("dev3", "W", None, 8))
    summary, groups = summarize_interval_coverage(
        calibrator, split, rows, {"dev1": 7.0, "dev2": 5.0, "dev3": 9.0},
        minimum_scaffold_n=3,
    )
    assert summary["n_labeled"] == 2
    assert summary["empirical_coverage"] == .5
    assert len(groups) == 1
    assert groups[0]["meets_minimum_n"] is False
    assert groups[0]["n"] == 2


def test_interval_rejects_nan_predictions():
    split, predictions = example_split()
    calibrated = fit_split_conformal(
        split, predictions, target="egfr", model_name="rf",
        expected_assignment_sha256=split.assignment_sha256,
    )
    with pytest.raises(ValueError, match="finite"):
        calibrated.intervals([float("nan")])


def test_interval_coverage_rejects_training_calibration_scaffolds():
    split, predictions = example_split()
    calibrator = fit_split_conformal(
        split, predictions, target="egfr", model_name="rf",
        expected_assignment_sha256=split.assignment_sha256,
    )
    with pytest.raises(ValueError, match="overlaps Stage C"):
        summarize_interval_coverage(calibrator, split, split.calibration,
                                    predictions)


def test_falsified_assignment_content_rejected_even_with_matching_claimed_hash():
    split, predictions = example_split()
    split.assignments = split.assignments[:-1] + (
        SimpleNamespace(structure_id="impostor", scaffold_group_id="T", role="threshold"),
    )
    with pytest.raises(ValueError, match="content does not match"):
        fit_split_conformal(split, predictions, target="egfr", model_name="rf",
                            expected_assignment_sha256=split.assignment_sha256)
