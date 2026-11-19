"""Small report-format tests for the Fold 0 engineering smoke driver."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from scripts.smoke_gnn_fold0 import epoch_rows, write_report


@dataclass
class FakeMetrics:
    egfr_mae: float
    her2_mae: float
    egfr_count: int
    her2_count: int

    @property
    def balanced_mae(self) -> float:
        return (self.egfr_mae + self.her2_mae) / 2


@dataclass
class FakeEpoch:
    epoch: int
    training_batch_loss: float
    stopping_metrics: FakeMetrics


def test_epoch_rows_preserve_both_target_metrics():
    history = (FakeEpoch(1, 0.8, FakeMetrics(0.6, 0.9, 5, 3)),)
    rows = epoch_rows(history)
    assert rows[0]["inner_balanced_mae"] == pytest.approx(0.75)
    assert rows[0]["inner_egfr_n"] == 5
    assert rows[0]["inner_her2_n"] == 3
    assert rows[0]["train_batch_loss"] == 0.8


def test_report_writes_json_and_aggregate_csv(tmp_path):
    report = {"cv_holdout_scored": False, "best_inner_epoch": 1}
    rows = epoch_rows((FakeEpoch(1, 0.2, FakeMetrics(0.5, 0.7, 9, 7)),))
    path = tmp_path / "subdir" / "smoke.json"
    write_report(path, report, rows)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["cv_holdout_scored"] is False
    assert loaded["best_inner_epoch"] == 1
    assert "inner_her2_mae" in (tmp_path / "subdir" / "smoke_epochs.csv").read_text(
        encoding="utf-8"
    )


def test_empty_epoch_history_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        write_report(tmp_path / "report.json", {}, [])
