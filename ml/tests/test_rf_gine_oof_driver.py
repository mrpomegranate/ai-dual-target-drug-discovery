"""Targeted tests for provenance and diagnostic summaries; no real data needed."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from scripts.compare_rf_gine_oof import (
    EXPECTED_BEST,
    build_scaffold_report,
    numeric_metrics,
    validate_study,
)


@dataclass(frozen=True)
class FakeBestTrial:
    number: int
    params: dict
    value: float


def make_study(target="egfr"):
    params = {key: value for key, value in EXPECTED_BEST[target].items() if key != "number"}
    return SimpleNamespace(
        user_attrs={
            "task": target,
            "metric_name": "mae",
            "dataset_sha256": "data-hash",
            "manifest_sha256": "manifest-hash",
            "feature_sha256": "fingerprint-hash",
            "fold_assignments_sha256": "fold-hash",
            "evaluation": "training_only_scaffold_group_cv",
            "reference_first": True,
            "base_config": {"radius": 2, "n_bits": 2048, "include_chirality": True,
                            "n_estimators": 200, "seed": 714, "n_jobs": -1},
        },
        best_trial=FakeBestTrial(EXPECTED_BEST[target]["number"], params, 0.7),
    )


@pytest.fixture
def context(monkeypatch):
    monkeypatch.setattr(
        "scripts.compare_rf_gine_oof.compute_fold_manifest_sha256", lambda _: "fold-hash"
    )
    pipeline = SimpleNamespace(
        data=SimpleNamespace(dataset_sha256="data-hash", manifest_sha256="manifest-hash")
    )
    bank = SimpleNamespace(fingerprint_sha256="fingerprint-hash")
    return pipeline, bank


@pytest.mark.parametrize("target", ["egfr", "her2"])
def test_reconstruct_selected_rf_config(target, context):
    pipeline, bank = context
    config = validate_study(make_study(target), target=target,
                            fit_only=pipeline, bank=bank)
    assert config.n_estimators == 200
    assert config.max_features == EXPECTED_BEST[target]["max_features"]
    assert config.seed == 714


def test_study_rejects_non_dict_base_config(context):
    pipeline, bank = context
    study = make_study()
    study.user_attrs["base_config"] = []

    with pytest.raises(TypeError, match="base RF configuration must be a dict"):
        validate_study(study, target="egfr", fit_only=pipeline, bank=bank)


@pytest.mark.parametrize("corruption,expected", [
    ("fingerprint", "feature_sha256"),
    ("folds", "fold_assignments_sha256"),
    ("trial", "best trial changed"),
    ("params", "parameters differ"),
    ("reference", "reference_first"),
])
def test_study_rejects_mismatch(corruption, expected, context):
    pipeline, bank = context
    study = make_study()
    if corruption == "fingerprint":
        study.user_attrs["feature_sha256"] = "wrong"
    elif corruption == "folds":
        study.user_attrs["fold_assignments_sha256"] = "wrong"
    elif corruption == "trial":
        study.best_trial = FakeBestTrial(4, study.best_trial.params, 0.7)
    elif corruption == "params":
        study.best_trial.params["max_features"] = 0.8
    else:
        study.user_attrs["reference_first"] = False
    with pytest.raises(ValueError, match=expected):
        validate_study(study, target="egfr", fit_only=pipeline, bank=bank)


def test_scaffold_report_preserves_missing_targets():
    rows = [
        {"scaffold_group_id": "S1", "egfr_mask": True, "her2_mask": False,
         "egfr_pic50": 7.0, "her2_pic50": None, "egfr_rf_pred": 6.5,
         "egfr_gnn_pred": 7.2, "her2_rf_pred": None, "her2_gnn_pred": 6.0},
        {"scaffold_group_id": "S1", "egfr_mask": False, "her2_mask": True,
         "egfr_pic50": None, "her2_pic50": 6.0, "egfr_rf_pred": None,
         "egfr_gnn_pred": 7.2, "her2_rf_pred": 6.8, "her2_gnn_pred": 6.3},
    ]
    result = build_scaffold_report(rows)
    assert len(result) == 1
    assert result[0]["egfr_n"] == 1 and result[0]["her2_n"] == 1
    assert result[0]["egfr_rf_mae"] == pytest.approx(0.5)
    assert result[0]["her2_gnn_mae"] == pytest.approx(0.3)


def test_numeric_metrics_excludes_none_and_booleans():
    result = numeric_metrics({"n": 5, "mae": 0.5, "invalid": None, "flag": True}, "rf_")
    assert result == {"rf_n": 5.0, "rf_mae": 0.5}
