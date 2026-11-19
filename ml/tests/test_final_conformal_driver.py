"""Pure protocol tests for M3.7b fit-role refit and calibration orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.fit_rf_gine_conformal import (
    EXPECTED_DATASET,
    EXPECTED_EPOCHS,
    EXPECTED_STAGE_C,
    make_ensemble,
    refuse_overwrite,
    select_final_epochs,
)


def _write_reference(path: Path, *, folds=None, **overrides) -> None:
    config = {
        "hidden_dim": 128, "num_layers": 3, "dropout": .1,
        "batch_size": 64, "learning_rate": .001, "weight_decay": .0001,
        "max_epochs": 80, "patience": 12, "seed": 719,
        "huber_delta": 1.0, "grad_clip_norm": 5.0,
    }
    source = {
        "dataset_sha256": EXPECTED_DATASET,
        "stage_c_assignment_sha256": EXPECTED_STAGE_C,
        "cv_holdout_scored": True,
        "calibration_scored": False,
        "threshold_scored": False,
        "development_validation_scored": False,
        "locked_test_scored": False,
        "config": config,
        "folds": [{"fold": i, "best_epoch": n} for i, n in enumerate(EXPECTED_EPOCHS)],
    }
    if folds is not None:
        source["folds"] = folds
    source.update(overrides)
    path.write_text(json.dumps(source), encoding="utf-8")


def test_freezes_median_inner_epoch(tmp_path):
    path = tmp_path / "cv.json"
    _write_reference(path)
    epoch, digest = select_final_epochs(path)
    assert epoch == 47 and len(digest) == 64


@pytest.mark.parametrize("change", [
    {"calibration_scored": True},
    {"development_validation_scored": True},
    {"locked_test_scored": True},
    {"dataset_sha256": "different"},
    {"stage_c_assignment_sha256": "different"},
    {"cv_holdout_scored": False},
])
def test_rejects_incompatible_reference(tmp_path, change):
    path = tmp_path / "cv.json"
    _write_reference(path, **change)
    with pytest.raises(ValueError):
        select_final_epochs(path)


def test_rejects_changed_epoch_selection(tmp_path):
    path = tmp_path / "cv.json"
    _write_reference(path, folds=[{"fold": i, "best_epoch": i+1} for i in range(5)])
    with pytest.raises(ValueError, match="epochs"):
        select_final_epochs(path)


def test_correct_fixed_ensemble():
    ensemble = make_ensemble({"a": 6., "b": 8.}, {"a": 8., "b": 4.}, {"a", "b"})
    assert ensemble == {"a": 6.5, "b": 7.0}


def test_reject_missing_ensemble_id():
    with pytest.raises(ValueError, match="precisely"):
        make_ensemble({"a": 6.}, {"a": 7., "b": 8.}, {"a", "b"})


def test_no_automatic_overwrite(tmp_path):
    file = tmp_path / "artifact.pt"
    file.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        refuse_overwrite([file], overwrite=False)
    refuse_overwrite([file], overwrite=True)
