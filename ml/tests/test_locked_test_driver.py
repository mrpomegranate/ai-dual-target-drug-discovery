"""M3.7d final protocol, no-overwrite and validation provenance tests."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts.evaluate_rf_gine_locked_test import (
    EXPECTED_DATASET,
    EXPECTED_MANIFEST,
    EXPECTED_STAGE_C,
    UNLOCK_TOKEN,
    build_frozen_protocol,
    marker_path,
    report_paths,
    require_one_time_authorization,
    reserve_one_time_test,
    validate_dev_report,
    verify_test_role,
)


def _calibration():
    return {
        "mlflow_run_id": "cal-run",
        "provenance": {"model_file_sha256": {
            "egfr_rf": "hash1", "her2_rf": "hash2", "gine": "hash3",
        }},
    }


def _dev(cal):
    return {
        "purpose": "independent_frozen_development_validation_no_test",
        "evaluation_role": "development_validation",
        "molecules": 1087,
        "models": ["rf", "gine", "rf75_gine25"],
        "nominal_marginal_coverage": .90,
        "locked_test_scored": False,
        "calibration_scored": False,
        "threshold_scored": False,
        "model_metrics": [
            {"model": m, "target": t}
            for t in ("egfr", "her2") for m in ("rf", "gine", "rf75_gine25")
        ],
        "provenance": {
            "dataset_sha256": EXPECTED_DATASET,
            "outer_manifest_sha256": EXPECTED_MANIFEST,
            "stage_c_assignment_sha256": EXPECTED_STAGE_C,
            "m3_7b_calibrator_report_sha256": "calhash",
            "model_file_sha256": cal["provenance"]["model_file_sha256"],
            "m3_7b_mlflow_run_id": "cal-run",
        },
    }


def test_valid_development_report():
    cal = _calibration()
    validate_dev_report(_dev(cal), cal, "calhash")


@pytest.mark.parametrize("mutator", [
    lambda d: d.update(locked_test_scored=True),
    lambda d: d.update(models=["rf", "gine"]),
    lambda d: d.update(molecules=100),
    lambda d: d["provenance"].update(m3_7b_calibrator_report_sha256="different"),
])
def test_dev_report_rejects_mismatch(mutator):
    cal = _calibration()
    dev = _dev(cal)
    mutator(dev)
    with pytest.raises(ValueError):
        validate_dev_report(dev, cal, "calhash")


def test_preflight_freeze_is_deterministic(tmp_path):
    paths = {}
    for key in ("dataset", "manifest", "summary", "calibrators", "dev_metrics", "egfr_rf", "her2_rf", "gine"):
        path = tmp_path / key
        path.write_bytes(key.encode())
        paths[key] = path
    kwargs = {
        "dataset": paths["dataset"], "manifest": paths["manifest"], "summary": paths["summary"],
        "calibrators": paths["calibrators"], "dev_metrics": paths["dev_metrics"],
        "model_paths": {key: paths[key] for key in ("egfr_rf", "her2_rf", "gine")},
        "minimum_scaffold_n": 10,
    }
    p1 = build_frozen_protocol(**kwargs)
    p2 = build_frozen_protocol(**kwargs)
    assert p1 == p2
    assert p1["model_configuration"]["ensemble_rf_weight"] == .75
    assert p1["evaluation"]["no_model_selection_from_test"] is True
    paths["gine"].write_bytes(b"mutated")
    assert build_frozen_protocol(**kwargs) != p1


def test_frozen_protocol_authorization_and_one_time_marker(tmp_path):
    outdir = tmp_path / "reports"
    outdir.mkdir()
    protocol_file = outdir / "frozen.json"
    protocol = {"final": True}
    protocol_file.write_text(json.dumps(protocol))
    sha = hashlib.sha256(protocol_file.read_bytes()).hexdigest()
    marker = marker_path(outdir)
    outputs = report_paths(outdir)
    kwargs = {"expected_sha": sha, "protocol_file": protocol_file,
                  "consumed_marker": marker, "outputs": outputs, "current_protocol": protocol}
    with pytest.raises(PermissionError):
        require_one_time_authorization(token="incorrect", **kwargs)
    with pytest.raises(ValueError, match="SHA-256"):
        require_one_time_authorization(token=UNLOCK_TOKEN, expected_sha="0" * 64,
                                       protocol_file=protocol_file,
                                       consumed_marker=marker, outputs=outputs,
                                       current_protocol=protocol)
    assert require_one_time_authorization(token=UNLOCK_TOKEN, **kwargs) == sha
    reserve_one_time_test(marker, sha, "run-id")
    assert json.loads(marker.read_text())["status"] == "test_unlock_started_do_not_rerun"
    with pytest.raises(FileExistsError):
        reserve_one_time_test(marker, sha, "second-run")
    with pytest.raises(FileExistsError):
        require_one_time_authorization(token=UNLOCK_TOKEN, **kwargs)


def test_changes_after_freeze_rejected(tmp_path):
    p = tmp_path / "protocol.json"
    p.write_text(json.dumps({"value": 1}))
    with pytest.raises(ValueError, match="changed after preflight"):
        require_one_time_authorization(
            token=UNLOCK_TOKEN,
            expected_sha=hashlib.sha256(p.read_bytes()).hexdigest(),
            protocol_file=p,
            consumed_marker=tmp_path / "unused",
            outputs={}, current_protocol={"value": 2},
        )


def test_no_outputs_may_preexist(tmp_path):
    p = tmp_path / "protocol.json"
    p.write_text(json.dumps({"v": 1}))
    output = tmp_path / "test_metrics.csv"
    output.touch()
    with pytest.raises(FileExistsError):
        require_one_time_authorization(
            token=UNLOCK_TOKEN,
            expected_sha=hashlib.sha256(p.read_bytes()).hexdigest(),
            protocol_file=p,
            consumed_marker=tmp_path / "unused",
            outputs={"metrics": output}, current_protocol={"v": 1},
        )


def test_scaffold_overlap_rejected():
    mk = lambda sid, sc: SimpleNamespace(structure_id=sid, scaffold_group_id=sc)
    source = SimpleNamespace(data=SimpleNamespace(train=(mk("tr", "same"),),
                                                validation=(mk("dev", "devscaf"),)))
    test = tuple([mk("t0", "same")] + [mk(f"t{i}", f"sc{i}") for i in range(1, 1087)])
    with pytest.raises(ValueError, match="overlaps"):
        verify_test_role(source, test)


def test_no_test_identity_overlap():
    mk = lambda sid, sc: SimpleNamespace(structure_id=sid, scaffold_group_id=sc)
    source = SimpleNamespace(data=SimpleNamespace(train=(mk("tr", "train"),),
                                                validation=(mk("dev", "devscaf"),)))
    test = tuple(mk(f"t{i}", f"sc{i}") for i in range(1087))
    verify_test_role(source, test)
