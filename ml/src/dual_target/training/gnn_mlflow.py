"""Optional MLflow evidence logging for the M3.4 GNN GPU engineering smoke.

This run is deliberately NOT a CV score or a selected production model.
Only frozen fit-only training/inner-stop statistics are logged. The refitted
model is stored as a reconstructible state_dict artifact, not registered.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from torch import nn

from ..features.molecular_graph import (
    ATOM_CARDINALITIES,
    ATOM_FEATURE_NAMES,
    BOND_CARDINALITIES,
    BOND_FEATURE_NAMES,
    GRAPH_SCHEMA_VERSION,
)

_RESERVED = (
    "cv_holdout_scored",
    "calibration_scored",
    "threshold_scored",
    "development_validation_scored",
    "locked_test_scored",
)


def _validate_smoke(report: Mapping) -> None:
    """Reject any report that includes non-smoke scoring or wrong graph schema."""
    if any(report.get(name) is not False for name in _RESERVED):
        raise ValueError("GNN smoke MLflow logger requires every reserved role unscored")
    if report.get("graph_schema_version") != GRAPH_SCHEMA_VERSION:
        raise ValueError("GNN graph schema does not match checkpoint featurizer")
    if report.get("fold") != 0 or report.get("fit_role_molecules") != 5364:
        raise ValueError("Unexpected frozen GNN smoke cohort")
    if not report.get("refit_training_batch_predictions_finite"):
        raise ValueError("Refit predictions were not verified finite")
    for key in (
        "dataset_sha256", "outer_manifest_sha256", "stage_c_assignment_sha256"
    ):
        if not report.get(key):
            raise ValueError(f"Missing provenance: {key}")


def save_smoke_checkpoint(model: nn.Module, report: Mapping, path: Path) -> Path:
    """Save CPU state and reconstructible architecture/schema metadata.

    PyTorch 2.10 can load this safe-primitive/tensor payload with
    ``torch.load(path, map_location='cpu', weights_only=True)``.
    """
    _validate_smoke(report)
    config = report["config"]
    model_config = {
        key: config[key] for key in ("hidden_dim", "num_layers", "dropout")
    }
    payload = {
        "state_dict": {
            name: weight.detach().cpu().clone()
            for name, weight in model.state_dict().items()
        },
        "model_config": model_config,
        "graph_schema_version": GRAPH_SCHEMA_VERSION,
        "atom_feature_names": tuple(ATOM_FEATURE_NAMES),
        "bond_feature_names": tuple(BOND_FEATURE_NAMES),
        "atom_cardinalities": tuple(ATOM_CARDINALITIES),
        "bond_cardinalities": tuple(BOND_CARDINALITIES),
        "dataset_sha256": report["dataset_sha256"],
        "outer_manifest_sha256": report["outer_manifest_sha256"],
        "stage_c_assignment_sha256": report["stage_c_assignment_sha256"],
        "fold": 0,
        "refit_epoch_count": report["refit_epoch_count"],
        "purpose": "engineering_smoke_only_not_selected_model",
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def log_smoke_run(
    *,
    report: Mapping,
    rows: Sequence[Mapping],
    report_path: Path,
    checkpoint_path: Path,
    tracking_uri: str,
    experiment_name: str,
) -> str:
    """Log smoke evidence and checkpoint to an EXPLICIT existing MLflow server.

    Intentionally no SQLite/default tracking fallback. A failed server request
    should fail visibly, not silently create a second experiment database.
    """
    _validate_smoke(report)
    if not rows:
        raise ValueError("Empty epoch history cannot be logged")
    if not tracking_uri.startswith(("http://", "https://")):
        raise ValueError("Use the existing HTTP(S) MLflow tracking server")
    for path in (report_path, checkpoint_path):
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="M3_4_GNN_Fold0_GPU_Engineering_Smoke") as run:
        mlflow.set_tags(
            {
                "model_family": "GNN",
                "stage": "M3.4",
                "run_type": "engineering_smoke_not_cv_score",
                "scaffold_protocol": "frozen_fit_only_seed719",
                "cv_holdout_scored": "false",
                "calibration_scored": "false",
                "threshold_scored": "false",
                "development_validation_scored": "false",
                "locked_test_scored": "false",
                "graph_schema_version": report["graph_schema_version"],
                "dataset_sha256": report["dataset_sha256"],
                "outer_manifest_sha256": report["outer_manifest_sha256"],
                "stage_c_assignment_sha256": report["stage_c_assignment_sha256"],
            }
        )
        mlflow.log_params({f"gnn_{k}": v for k, v in report["config"].items()})
        mlflow.log_params(
            {
                "smoke_fold": report["fold"],
                "fold_fit_molecules": report["fold_fit_molecules"],
                "inner_fit_molecules": report["inner_fit_molecules"],
                "inner_stop_molecules": report["inner_stop_molecules"],
                "refit_epoch_count": report["refit_epoch_count"],
                "torch_version": report["torch_version"],
                "device": report["device"],
            }
        )
        for row in rows:
            epoch = int(row["epoch"])
            for name in (
                "train_batch_loss", "inner_egfr_mae", "inner_her2_mae",
                "inner_balanced_mae",
            ):
                value = float(row[name])
                if not math.isfinite(value):
                    raise ValueError(f"Nonfinite {name} at epoch {epoch}")
                mlflow.log_metric(f"smoke_{name}", value, step=epoch)
        mlflow.log_metrics(
            {
                "smoke_best_inner_egfr_mae": report["best_inner_egfr_mae"],
                "smoke_best_inner_her2_mae": report["best_inner_her2_mae"],
                "smoke_best_inner_balanced_mae": report["best_inner_balanced_mae"],
                "smoke_peak_gpu_memory_mib": report["peak_gpu_memory_mib"] or 0.0,
            }
        )
        mlflow.log_artifact(str(report_path), artifact_path="evidence")
        csv_path = Path(report["epoch_history_csv"])
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        mlflow.log_artifact(str(csv_path), artifact_path="evidence")
        mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")
        return run.info.run_id
