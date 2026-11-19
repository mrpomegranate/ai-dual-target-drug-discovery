"""GNN smoke checkpoint and MLflow tests without any network requests."""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from dual_target.features.molecular_graph import GRAPH_SCHEMA_VERSION
from dual_target.training.gnn_mlflow import log_smoke_run, save_smoke_checkpoint
from torch import nn


def example_report():
    return {
        "cv_holdout_scored": False,
        "calibration_scored": False,
        "threshold_scored": False,
        "development_validation_scored": False,
        "locked_test_scored": False,
        "fold": 0,
        "fit_role_molecules": 5364,
        "fold_fit_molecules": 4291,
        "inner_fit_molecules": 3285,
        "inner_stop_molecules": 1006,
        "refit_training_batch_predictions_finite": True,
        "refit_epoch_count": 2,
        "graph_schema_version": GRAPH_SCHEMA_VERSION,
        "dataset_sha256": "dataset-hash",
        "outer_manifest_sha256": "manifest-hash",
        "stage_c_assignment_sha256": "assignment-hash",
        "config": {"hidden_dim": 16, "num_layers": 1, "dropout": 0.0},
        "torch_version": torch.__version__,
        "device": "cpu",
        "best_inner_egfr_mae": 0.8,
        "best_inner_her2_mae": 0.9,
        "best_inner_balanced_mae": 0.85,
        "peak_gpu_memory_mib": None,
    }


def test_checkpoint_is_weights_only_loadable_and_contains_schema(tmp_path):
    model = nn.Linear(2, 1)
    path = save_smoke_checkpoint(model, example_report(), tmp_path / "weights.pt")
    recovered = torch.load(path, map_location="cpu", weights_only=True)
    assert recovered["purpose"] == "engineering_smoke_only_not_selected_model"
    assert recovered["graph_schema_version"] == GRAPH_SCHEMA_VERSION
    assert recovered["model_config"]["hidden_dim"] == 16
    assert set(recovered["state_dict"]) == set(model.state_dict())
    assert torch.equal(recovered["state_dict"]["weight"], model.weight.data)


def test_logger_rejects_scored_holdout_and_invalid_uri(tmp_path):
    report = example_report()
    report["cv_holdout_scored"] = True
    with pytest.raises(ValueError, match="reserved role"):
        save_smoke_checkpoint(nn.Linear(2, 1), report, tmp_path / "bad.pt")
    report = example_report()
    file = tmp_path / "exists.txt"
    file.write_text("example")
    with pytest.raises(ValueError, match="HTTP"):
        log_smoke_run(
            report=report, rows=[{"epoch": 1}],
            report_path=file, checkpoint_path=file, tracking_uri="sqlite:///local.db",
            experiment_name="example",
        )


def test_mlflow_logs_evidence_checkpoint_and_metrics_without_network(tmp_path, monkeypatch):
    report = example_report()
    report_path = tmp_path / "smoke.json"
    report_path.write_text(json.dumps(report))
    checkpoint = tmp_path / "smoke.pt"
    checkpoint.write_bytes(b"fake")
    csv_path = tmp_path / "smoke_epochs.csv"
    csv_path.write_text("epoch,train_batch_loss,inner_egfr_mae,inner_her2_mae,inner_balanced_mae\n")
    report["epoch_history_csv"] = str(csv_path)
    recorded = []

    @contextmanager
    def fake_run(**kwargs):
        recorded.append(("start_run", kwargs))
        yield SimpleNamespace(info=SimpleNamespace(run_id="mock-run"))

    fake = SimpleNamespace(
        set_tracking_uri=lambda *a, **k: recorded.append(("set_tracking_uri", a, k)),
        set_experiment=lambda *a, **k: recorded.append(("set_experiment", a, k)),
        start_run=fake_run,
        set_tags=lambda *a, **k: recorded.append(("set_tags", a, k)),
        log_params=lambda *a, **k: recorded.append(("log_params", a, k)),
        log_metric=lambda *a, **k: recorded.append(("log_metric", a, k)),
        log_metrics=lambda *a, **k: recorded.append(("log_metrics", a, k)),
        log_artifact=lambda *a, **k: recorded.append(("log_artifact", a, k)),
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    run_id = log_smoke_run(
        report=report,
        rows=[{
            "epoch": 1, "train_batch_loss": 1.2,
            "inner_egfr_mae": 0.8, "inner_her2_mae": 0.9,
            "inner_balanced_mae": 0.85,
        }],
        report_path=report_path, checkpoint_path=checkpoint,
        tracking_uri="http://127.0.0.1:5000", experiment_name="existing-experiment",
    )
    assert run_id == "mock-run"
    assert sum(name == "log_artifact" for name, *_ in recorded) == 3
    assert sum(name == "log_metric" for name, *_ in recorded) == 4
    assert any(name == "set_experiment" for name, *_ in recorded)
