"""M3.7c frozen-checkpoint and development-validation role protections."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from scripts.evaluate_rf_gine_dev_validation import (
    EXPECTED_DATASET,
    EXPECTED_MANIFEST,
    EXPECTED_STAGE_C,
    make_validation_frame,
    verify_frozen_report,
    verify_validation_role,
)


def _report(tmp_path):
    paths = {}
    hashes = {}
    for name in ("egfr_rf", "her2_rf", "gine"):
        path = tmp_path / f"{name}.chk"
        path.write_bytes(name.encode())
        paths[name] = path
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    calibrators = {
        model: {
            target: {
                "model_name": model, "target": target, "alpha": .1,
                "absolute_error_quantile": .8, "n_labels": 90, "n_scaffolds": 20,
                "stage_c_assignment_sha256": EXPECTED_STAGE_C,
                "dataset_sha256": EXPECTED_DATASET,
                "outer_manifest_sha256": EXPECTED_MANIFEST,
            }
            for target in ("egfr", "her2")
        }
        for model in ("rf", "gine", "rf75_gine25")
    }
    return {
        "purpose": "fit_role_refit_and_frozen_stage_c_conformal_calibration",
        "n_fit": 5364, "n_calibration": 1658, "n_threshold_unscored": 1670,
        "ensemble_rf_weight": .75, "alpha": .1,
        "training": {"gine_epochs": 47},
        "calibration_residuals_used": True,
        "threshold_scored": False, "development_validation_scored": False,
        "locked_test_scored": False,
        "provenance": {
            "dataset_sha256": EXPECTED_DATASET,
            "outer_manifest_sha256": EXPECTED_MANIFEST,
            "stage_c_assignment_sha256": EXPECTED_STAGE_C,
            "graph_schema_version": "m3_molecular_graph_v1",
            "model_file_sha256": hashes,
        },
        "calibrators": calibrators,
    }, paths


def test_valid_calibration_report(tmp_path):
    report, paths = _report(tmp_path)
    verify_frozen_report(report, model_paths=paths)


@pytest.mark.parametrize("key,value", [
    ("ensemble_rf_weight", .50), ("alpha", .05),
    ("threshold_scored", True), ("locked_test_scored", True),
    ("development_validation_scored", True),
])
def test_refuses_changed_recipe_or_used_reserved_labels(tmp_path, key, value):
    report, paths = _report(tmp_path)
    report[key] = value
    with pytest.raises(ValueError):
        verify_frozen_report(report, model_paths=paths)


def test_detects_mutated_model_checkpoint(tmp_path):
    report, paths = _report(tmp_path)
    paths["gine"].write_bytes(b"changed")
    with pytest.raises(ValueError, match="Checkpoint"):
        verify_frozen_report(report, model_paths=paths)


def test_detects_calibrator_provenance_mismatch(tmp_path):
    report, paths = _report(tmp_path)
    report["calibrators"]["gine"]["egfr"]["dataset_sha256"] = "wrong"
    with pytest.raises(ValueError, match="Calibrator provenance"):
        verify_frozen_report(report, model_paths=paths)


def test_detects_invalid_calibrator_quantile(tmp_path):
    report, paths = _report(tmp_path)
    report["calibrators"]["rf"]["her2"]["absolute_error_quantile"] = -1
    with pytest.raises(ValueError, match="quantile"):
        verify_frozen_report(report, model_paths=paths)


def test_validation_role_rejects_wrong_size():
    rec = lambda sid, sc: SimpleNamespace(structure_id=sid, scaffold_group_id=sc)
    split = SimpleNamespace(fit=(rec("fit", "sfit"),), calibration=(), threshold=())
    with pytest.raises(ValueError, match="1,087"):
        verify_validation_role(split.fit, (rec("v", "sv"),), split)


def test_validated_frame_uses_fixed_ensemble():
    rec = SimpleNamespace(structure_id="a", scaffold_group_id="s1",
                          egfr_mask=1, her2_mask=0, egfr_pic50=6., her2_pic50=None)
    rf = {"egfr": {"a": 6.}, "her2": {"a": 7.}}
    gine = {"egfr": {"a": 8.}, "her2": {"a": 5.}}
    frame = make_validation_frame((rec,), rf, gine)
    assert frame.egfr_rf75_gine25_pred.iloc[0] == 6.5
    assert frame.her2_rf75_gine25_pred.iloc[0] == 6.5


def test_frame_rejects_missing_predictions():
    rec = SimpleNamespace(structure_id="a", scaffold_group_id="s1",
                          egfr_mask=1, her2_mask=1, egfr_pic50=6., her2_pic50=6.)
    rf = {"egfr": {"a": 6.}, "her2": {"a": 6.}}
    gine = {"egfr": {}, "her2": {"a": 5.}}
    with pytest.raises(ValueError, match="Missing or extra"):
        make_validation_frame((rec,), rf, gine)
