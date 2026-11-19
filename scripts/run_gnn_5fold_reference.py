"""Fit-only 5-fold GNN *reference* CV; no reserved-role evaluation.

One prespecified GNN configuration. Each existing scaffold fold uses an inner
scaffold-disjoint stopping subset from its CV-fit portion. The fresh refit is
then scored on that fold's held-out scaffolds. This is model-development CV:
its results are NOT a pristine final performance estimate after tuning.
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
    make_graph_loader,
    select_fold_graphs,
)
from dual_target.data.pipeline import (
    FROZEN_STAGE_C_ASSIGNMENT_SHA256,
    build_fit_only_tuning_pipeline,
    load_modeling_pipeline,
)
from dual_target.evaluation.gnn_cv_metrics import score_oof_rows, verify_oof_coverage
from dual_target.features.molecular_graph import (
    ATOM_CARDINALITIES,
    ATOM_FEATURE_NAMES,
    BOND_CARDINALITIES,
    BOND_FEATURE_NAMES,
    GRAPH_SCHEMA_VERSION,
)
from dual_target.training.gnn_training import (
    GNNTrainingConfig,
    refit_cv_training_graphs,
    train_with_inner_stopping,
)

DATASET_HASH = "83a64d02e45ccd9a6310947efcd5cdef9bddc9603edcd55e1d207eb1cafc18b8"
EXPERIMENT = "EGFR_HER2_M2_Scaffold_Benchmark"


def predict_oof(model, graphs, *, fold: int, batch_size: int, device) -> list[dict]:
    """Score exactly the provided fold holdout after fixed-epoch refitting."""
    model.eval()
    rows = []
    with torch.inference_mode():
        for cpu_batch in make_graph_loader(graphs, batch_size=batch_size, shuffle=False):
            ids = list(cpu_batch.structure_id)
            labels = cpu_batch.y.detach().cpu().tolist()
            masks = cpu_batch.y_mask.detach().cpu().tolist()
            predictions = model(cpu_batch.to(device)).detach().cpu().tolist()
            if not (len(ids) == len(labels) == len(masks) == len(predictions)):
                raise ValueError("Batch identity, label, mask, prediction mismatch")
            for sid, actual, mask, pred in zip(ids, labels, masks, predictions):
                if not all(math.isfinite(float(x)) for x in pred):
                    raise ValueError("Nonfinite CV holdout prediction")
                rows.append({
                    "structure_id": sid,
                    "fold": fold,
                    "egfr_mask": bool(mask[0]),
                    "her2_mask": bool(mask[1]),
                    "egfr_pic50": float(actual[0]) if mask[0] else None,
                    "her2_pic50": float(actual[1]) if mask[1] else None,
                    "egfr_pred": float(pred[0]),
                    "her2_pred": float(pred[1]),
                })
    if len(rows) != len(graphs):
        raise ValueError("CV holdout prediction count mismatch")
    if {row["structure_id"] for row in rows} != {g.structure_id for g in graphs}:
        raise ValueError("CV holdout prediction identity mismatch")
    return rows


def save_checkpoint(model, *, fold: int, epochs: int, config, bank, path: Path) -> None:
    """Store an unregistered, reloadable state_dict and schema metadata."""
    payload = {
        "state_dict": {name: weight.detach().cpu().clone()
                       for name, weight in model.state_dict().items()},
        "model_config": {name: getattr(config, name)
                         for name in ("hidden_dim", "num_layers", "dropout")},
        "training_config": asdict(config),
        "graph_schema_version": GRAPH_SCHEMA_VERSION,
        "atom_feature_names": tuple(ATOM_FEATURE_NAMES),
        "bond_feature_names": tuple(BOND_FEATURE_NAMES),
        "atom_cardinalities": tuple(ATOM_CARDINALITIES),
        "bond_cardinalities": tuple(BOND_CARDINALITIES),
        "dataset_sha256": bank.dataset_sha256,
        "outer_manifest_sha256": bank.manifest_sha256,
        "stage_c_assignment_sha256": bank.stage_c_assignment_sha256,
        "fold": fold,
        "refit_epoch_count": epochs,
        "purpose": "fit_only_reference_cv_not_final_model",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Cannot write empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def log_numeric_metrics(mlflow, metrics: dict, prefix: str) -> None:
    mlflow.log_metrics({f"{prefix}{key}": float(value) for key, value in metrics.items()
                        if isinstance(value, (int, float)) and math.isfinite(value)})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default=Path("data/interim/M2_1_scaffold_split_molecules.csv"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/interim/M2_1_scaffold_split_manifest.csv"))
    parser.add_argument("--summary", type=Path,
                        default=Path("reports/audits/M2_1_scaffold_split_summary.json"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/milestones"))
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--mlflow-uri", default="http://127.0.0.1:5000")
    parser.add_argument("--mlflow-experiment", default=EXPERIMENT)
    args = parser.parse_args(argv)
    if args.max_epochs < 20:
        parser.error("Use at least 20 max epochs for reference CV; 3-epoch engineering smoke is separate")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    if not args.mlflow_uri.startswith(("http://", "https://")):
        parser.error("Explicit HTTP(S) MLflow server required; no local DB fallback")
    config = replace(GNNTrainingConfig(), max_epochs=args.max_epochs,
                     patience=args.patience)
    source = load_modeling_pipeline(args.dataset, args.manifest, args.summary)
    if source.data.dataset_sha256 != DATASET_HASH:
        raise ValueError("Frozen dataset hash mismatch")
    split = make_scaffold_calibration_split(
        source, seed=719, calibration_fraction=0.20, threshold_fraction=0.20,
        minimum_target_labels=10, minimum_dual_per_class=50, max_attempts=1,
    )
    fit_only = build_fit_only_tuning_pipeline(source, split, n_splits=5)
    if fit_only.stage_c_assignment_sha256 != FROZEN_STAGE_C_ASSIGNMENT_SHA256:
        raise ValueError("Frozen Stage C assignment hash mismatch")
    bank = build_fit_only_graph_bank(fit_only)
    if bank.molecules != 5364 or len(fit_only.folds) != 5:
        raise ValueError("Unexpected fit-only molecules or CV folds")

    # Always log to existing HTTP tracking backend, never create another SQLite DB.
    import mlflow

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.mlflow_experiment)
    all_rows: list[dict] = []
    fold_reports: list[dict] = []
    args.report_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    with mlflow.start_run(run_name="M3_5_GNN_Reference_5Fold_CV") as parent:
        mlflow.set_tags({
            "stage": "M3.5", "model_family": "GNN", "cv_type": "fit_only_scaffold_5fold",
            "run_type": "prespecified_reference_cv", "cv_holdout_scored": "true",
            "development_validation_scored": "false",
            "calibration_scored": "false", "threshold_scored": "false",
            "locked_test_scored": "false", "dataset_sha256": bank.dataset_sha256,
            "outer_manifest_sha256": bank.manifest_sha256,
            "stage_c_assignment_sha256": bank.stage_c_assignment_sha256,
            "graph_schema_version": bank.graph_schema_version,
        })
        mlflow.log_params({f"gnn_{k}": v for k, v in asdict(config).items()})
        mlflow.log_param("device", args.device)
        print(f"MLflow parent run: {parent.info.run_id}")
        for fold_number in range(5):
            view = select_fold_graphs(fit_only, bank, fold_number)
            print(f"Fold {fold_number}: inner-stop selection on {len(view.fit)} CV-fit graphs")
            result = train_with_inner_stopping(view, config=config, device=device)
            model = refit_cv_training_graphs(
                view, epochs=result.best_epoch, config=config, device=device,
            )
            # CV holdout is touched for scoring ONLY after selection and refit.
            rows = predict_oof(
                model, view.holdout, fold=fold_number,
                batch_size=config.batch_size, device=device,
            )
            metrics = score_oof_rows(rows)
            entry = {
                "fold": fold_number,
                "fit_n": len(view.fit),
                "holdout_n": len(view.holdout),
                "inner_fit_n": result.inner_train_molecules,
                "inner_stop_n": result.inner_stop_molecules,
                "best_epoch": result.best_epoch,
                "epochs_ran": result.epochs_ran,
                **metrics,
            }
            fold_reports.append(entry)
            all_rows.extend(rows)
            checkpoint_path = args.model_dir / f"M3_5_GNN_reference_fold{fold_number}.pt"
            save_checkpoint(
                model, fold=fold_number, epochs=result.best_epoch,
                config=config, bank=bank, path=checkpoint_path,
            )
            with mlflow.start_run(run_name=f"M3_5_reference_fold_{fold_number}", nested=True):
                mlflow.set_tags({"fold": str(fold_number), "role": "cv_holdout_score"})
                mlflow.log_params({
                    "cv_fit_n": len(view.fit), "cv_holdout_n": len(view.holdout),
                    "best_inner_epoch": result.best_epoch,
                    "epochs_ran": result.epochs_ran,
                })
                log_numeric_metrics(mlflow, metrics, "cv_")
                for epoch in result.history:
                    mlflow.log_metric("inner_train_loss", epoch.training_batch_loss,
                                      step=epoch.epoch)
                    mlflow.log_metric("inner_stop_egfr_mae", epoch.stopping_metrics.egfr_mae,
                                      step=epoch.epoch)
                    mlflow.log_metric("inner_stop_her2_mae", epoch.stopping_metrics.her2_mae,
                                      step=epoch.epoch)
                mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")
            print(f"  MAE EGFR={metrics['egfr_mae']:.4f}, HER2={metrics['her2_mae']:.4f}; "
                  f"Spearman EGFR={metrics['egfr_spearman']}, HER2={metrics['her2_spearman']}; "
                  f"paired AP={metrics['dual_min_prediction_ap']}; "
                  f"selected epoch={result.best_epoch}")

        verify_oof_coverage(all_rows, bank.ordered_ids, expected_folds=5)
        pooled = score_oof_rows(all_rows)
        report = {
            "purpose": "Prespecified fit-only GNN reference fivefold CV",
            "model_selected": False,
            "cv_holdout_scored": True,
            "calibration_scored": False,
            "threshold_scored": False,
            "development_validation_scored": False,
            "locked_test_scored": False,
            "config": asdict(config),
            "torch_version": torch.__version__,
            "device": args.device,
            "graph_schema_version": bank.graph_schema_version,
            "dataset_sha256": bank.dataset_sha256,
            "outer_manifest_sha256": bank.manifest_sha256,
            "stage_c_assignment_sha256": bank.stage_c_assignment_sha256,
            "mlflow_parent_run_id": parent.info.run_id,
            "folds": fold_reports,
            "pooled_oof": pooled,
        }
        out_json = args.report_dir / "M3_5_GNN_reference_5fold_cv.json"
        out_csv = args.report_dir / "M3_5_GNN_reference_5fold_metrics.csv"
        out_oof = args.report_dir / "M3_5_GNN_reference_oof_predictions.csv"
        out_json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        write_csv(out_csv, fold_reports)
        write_csv(out_oof, all_rows)
        log_numeric_metrics(mlflow, pooled, "pooled_oof_")
        for path in (out_json, out_csv, out_oof):
            mlflow.log_artifact(str(path), artifact_path="evidence")
        print(f"Pooled fit-only OOF EGFR/HER2 MAE: {pooled['egfr_mae']:.4f} / "
              f"{pooled['her2_mae']:.4f}")
        print(f"Pooled paired AP from min(predicted pIC50): {pooled['dual_min_prediction_ap']}")
        print(f"Saved: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
