"""Run a fit-only Fold 0 GNN GPU smoke test without CV-holdout scoring.

This is an engineering check, NOT model selection or a five-fold CV score.
The script only trains on the existing Fold 0 .fit graphs, uses an inner
scaffold-disjoint stop subset to choose an epoch, and checks a fresh refit.
It never evaluates Fold 0's designated .holdout labels or reserved roles.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import torch
from dual_target.calibration.scaffold_calibration import make_scaffold_calibration_split
from dual_target.data.gnn_dataset import (
    build_fit_only_graph_bank,
    select_fold_graphs,
)
from dual_target.data.pipeline import (
    FROZEN_STAGE_C_ASSIGNMENT_SHA256,
    build_fit_only_tuning_pipeline,
    load_modeling_pipeline,
)
from dual_target.training.gnn_mlflow import log_smoke_run, save_smoke_checkpoint
from dual_target.training.gnn_training import (
    GNNTrainingConfig,
    refit_cv_training_graphs,
    train_with_inner_stopping,
)
from torch_geometric.loader import DataLoader

EXPECTED_DATASET_SHA256 = (
    "83a64d02e45ccd9a6310947efcd5cdef9bddc9603edcd55e1d207eb1cafc18b8"
)


def epoch_rows(history) -> list[dict[str, int | float]]:
    """Produce an audit-friendly, aggregated epoch history (no raw labels)."""
    rows = []
    for entry in history:
        stop = entry.stopping_metrics
        rows.append(
            {
                "epoch": int(entry.epoch),
                "train_batch_loss": float(entry.training_batch_loss),
                "inner_egfr_mae": float(stop.egfr_mae),
                "inner_her2_mae": float(stop.her2_mae),
                "inner_balanced_mae": float(stop.balanced_mae),
                "inner_egfr_n": int(stop.egfr_count),
                "inner_her2_n": int(stop.her2_count),
            }
        )
    return rows


def write_report(output: Path, report: dict, rows: list[dict]) -> None:
    """Write only aggregate inner-stop diagnostics and immutable provenance."""
    if not rows:
        raise ValueError("Cannot report an empty training history")
    output.parent.mkdir(parents=True, exist_ok=True)
    history_csv = output.with_name(f"{output.stem}_epochs.csv")
    with history_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report["epoch_history_csv"] = str(history_csv)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path,
        default=Path("data/interim/M2_1_scaffold_split_molecules.csv"),
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("data/interim/M2_1_scaffold_split_manifest.csv"),
    )
    parser.add_argument(
        "--summary", type=Path,
        default=Path("reports/audits/M2_1_scaffold_split_summary.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("reports/milestones/M3_4_fold0_gpu_smoke.json"),
    )
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--log-mlflow", action="store_true",
        help="Log this engineering smoke run and refit checkpoint to the existing MLflow server",
    )
    parser.add_argument(
        "--mlflow-uri", default="http://127.0.0.1:5000",
        help="Existing MLflow tracking server; never silently create another local store",
    )
    parser.add_argument(
        "--mlflow-experiment", default="EGFR_HER2_M2_Scaffold_Benchmark",
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("artifacts/models/M3_4_fold0_smoke_refit_state_dict.pt"),
    )
    args = parser.parse_args(argv)

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available; use --device cpu for a CPU-only check")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    source = load_modeling_pipeline(args.dataset, args.manifest, args.summary)
    if source.data.dataset_sha256 != EXPECTED_DATASET_SHA256:
        raise ValueError("Frozen dataset checksum differs from M2.1 reference")
    split = make_scaffold_calibration_split(
        source,
        seed=719,
        calibration_fraction=0.20,
        threshold_fraction=0.20,
        minimum_target_labels=10,
        minimum_dual_per_class=50,
        max_attempts=1,
    )
    fit_only = build_fit_only_tuning_pipeline(source, split, n_splits=5)
    if fit_only.stage_c_assignment_sha256 != FROZEN_STAGE_C_ASSIGNMENT_SHA256:
        raise ValueError("Stage C frozen assignment checksum mismatch")
    graph_bank = build_fit_only_graph_bank(fit_only)
    if graph_bank.molecules != 5364:
        raise ValueError("Unexpected number of fit-role molecules")
    fold0 = select_fold_graphs(fit_only, graph_bank, fold_number=0)
    if len(fold0.fit) != 4291 or len(fold0.holdout) != 1073:
        raise ValueError("Unexpected frozen Fold 0 partition size")

    config = replace(
        GNNTrainingConfig(),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )
    print(f"Fold 0: {len(fold0.fit)} training graphs, "
          f"{len(fold0.holdout)} CV-holdout graphs (NOT scored)")
    print(f"Device: {device}; inner stopping and fresh CV-fit refit only")
    result = train_with_inner_stopping(fold0, config=config, device=device)
    refit = refit_cv_training_graphs(
        fold0, epochs=result.best_epoch, config=config, device=device
    )

    # Engineering-only forward pass on a CV-TRAINING mini-batch.
    train_batch = next(iter(DataLoader(list(fold0.fit[:config.batch_size]),
                                       batch_size=config.batch_size)))
    refit.eval()
    with torch.inference_mode():
        predictions = refit(train_batch.to(device))
    if predictions.shape != (train_batch.num_graphs, 2):
        raise ValueError("Fresh refit produced unexpected prediction shape")
    if not bool(torch.isfinite(predictions).all()):
        raise ValueError("Fresh refit produced nonfinite potency predictions")

    rows = epoch_rows(result.history)
    if any(not math.isfinite(float(row[k])) for row in rows for k in (
        "train_batch_loss", "inner_egfr_mae", "inner_her2_mae",
        "inner_balanced_mae",
    )):
        raise ValueError("Training history contains nonfinite metrics")
    report = {
        "purpose": "Fold 0 GPU engineering smoke test, NOT CV model selection",
        "cv_holdout_scored": False,
        "calibration_scored": False,
        "threshold_scored": False,
        "development_validation_scored": False,
        "locked_test_scored": False,
        "fold": 0,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_gpu_memory_mib": (
            round(torch.cuda.max_memory_allocated(device) / 2**20, 1)
            if device.type == "cuda" else None
        ),
        "torch_version": torch.__version__,
        "config": asdict(config),
        "fit_role_molecules": graph_bank.molecules,
        "fold_fit_molecules": len(fold0.fit),
        "fold_holdout_molecules_not_scored": len(fold0.holdout),
        "inner_fit_molecules": result.inner_train_molecules,
        "inner_stop_molecules": result.inner_stop_molecules,
        "inner_fit_scaffolds": result.inner_train_scaffolds,
        "inner_stop_scaffolds": result.inner_stop_scaffolds,
        "epochs_ran": result.epochs_ran,
        "best_inner_epoch": result.best_epoch,
        "best_inner_balanced_mae": result.best_inner_mae,
        "best_inner_egfr_mae": rows[result.best_epoch - 1]["inner_egfr_mae"],
        "best_inner_her2_mae": rows[result.best_epoch - 1]["inner_her2_mae"],
        "refit_epoch_count": result.best_epoch,
        "refit_training_batch_predictions_finite": True,
        "refit_training_batch_shape": list(predictions.shape),
        "dataset_sha256": graph_bank.dataset_sha256,
        "outer_manifest_sha256": graph_bank.manifest_sha256,
        "stage_c_assignment_sha256": graph_bank.stage_c_assignment_sha256,
        "graph_schema_version": graph_bank.graph_schema_version,
    }
    write_report(args.output, report, rows)
    if args.log_mlflow:
        save_smoke_checkpoint(refit, report, args.checkpoint)
        run_id = log_smoke_run(
            report=report, rows=rows, report_path=args.output,
            checkpoint_path=args.checkpoint, tracking_uri=args.mlflow_uri,
            experiment_name=args.mlflow_experiment,
        )
        print(f"MLflow run ID: {run_id}")
        print(f"Saved checkpoint: {args.checkpoint}")
    print(f"Best inner stopping epoch: {result.best_epoch}/{result.epochs_ran}")
    print(f"Inner-stop EGFR/HER2 MAE: {report['best_inner_egfr_mae']:.4f} / "
          f"{report['best_inner_her2_mae']:.4f}")
    print(f"Fresh refit completed ({result.best_epoch} epochs; training batch finite)")
    print(f"Saved report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
