"""Edge-aware GATv2 reference CV on the frozen Stage C fit-only folds.

Uses the exact five scaffold folds previously scored with GINE. For each fold,
selects the stopping epoch on an inner scaffold split of CV-fit molecules, refits
on all CV-fit molecules, then scores that fold's held-out scaffolds exactly once.

This is development CV, not a pristine post-model-selection test. No reserved
calibration, threshold, dev-validation, or locked-test labels are scored.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
from dual_target.models.gat import MolecularGATv2
from dual_target.training.gnn_training import (
    GNNTrainingConfig,
    refit_cv_training_graphs,
    train_with_inner_stopping,
)

DATASET_HASH = "83a64d02e45ccd9a6310947efcd5cdef9bddc9603edcd55e1d207eb1cafc18b8"
EXPERIMENT = "EGFR_HER2_M2_Scaffold_Benchmark"
PREFIX = "M3_6_GATv2_reference"
REFERENCE_OOF = Path("reports/milestones/M3_5_GNN_reference_oof_predictions.csv")


def read_reference_fold_membership(path: Path) -> dict[str, int]:
    """Read only identity and fold membership from existing fit-only GINE OOF."""
    mapping: dict[str, int] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"structure_id", "fold"}.issubset(reader.fieldnames or ()):
            raise ValueError("GINE OOF file lacks structure_id/fold columns")
        for row in reader:
            sid = row["structure_id"]
            if not sid or sid in mapping:
                raise ValueError("GINE OOF contains missing or repeated molecule ID")
            try:
                number = int(row["fold"])
            except (ValueError, TypeError) as error:
                raise ValueError("GINE OOF contains invalid fold ID") from error
            if number not in range(5):
                raise ValueError("GINE OOF contains out-of-range fold ID")
            mapping[sid] = number
    if not mapping:
        raise ValueError("GINE OOF is empty")
    return mapping


def verify_reference_fold_membership(fit_only, mapping: dict[str, int]) -> None:
    """Require the saved GINE OOF to match every frozen GATv2 CV assignment."""
    current = {
        record.structure_id: fold.number
        for fold in fit_only.folds
        for record in fold.holdout
    }
    if current != mapping:
        missing = set(current) - set(mapping)
        extra = set(mapping) - set(current)
        changed = {sid for sid in set(current) & set(mapping) if current[sid] != mapping[sid]}
        raise ValueError(
            "GINE reference OOF fold membership differs from frozen GATv2 folds: "
            f"missing={len(missing)}, extra={len(extra)}, changed={len(changed)}"
        )


def predict_oof(model, graphs, *, fold: int, batch_size: int, device) -> list[dict]:
    """Score the current designated CV holdout once, after final refit."""
    model.eval()
    rows: list[dict] = []
    with torch.inference_mode():
        for cpu_batch in make_graph_loader(graphs, batch_size=batch_size, shuffle=False):
            ids = list(cpu_batch.structure_id)
            labels = cpu_batch.y.detach().cpu().tolist()
            masks = cpu_batch.y_mask.detach().cpu().tolist()
            preds = model(cpu_batch.to(device)).detach().cpu().tolist()
            if not (len(ids) == len(labels) == len(masks) == len(preds)):
                raise ValueError("OOF batch identities, targets, masks or predictions misaligned")
            for sid, observed, mask, predicted in zip(ids, labels, masks, preds, strict=True):
                if not all(math.isfinite(float(value)) for value in predicted):
                    raise ValueError(f"Nonfinite prediction for {sid}")
                rows.append({
                    "structure_id": sid,
                    "fold": fold,
                    "egfr_mask": bool(mask[0]),
                    "her2_mask": bool(mask[1]),
                    "egfr_pic50": float(observed[0]) if mask[0] else None,
                    "her2_pic50": float(observed[1]) if mask[1] else None,
                    "egfr_pred": float(predicted[0]),
                    "her2_pred": float(predicted[1]),
                })
    if len(rows) != len(graphs) or {row["structure_id"] for row in rows} != {
        graph.structure_id for graph in graphs
    }:
        raise ValueError("CV holdout OOF coverage differs from input graph identities")
    return rows


def epoch_rows(history) -> list[dict]:
    """Aggregate inner-training losses and stopping metrics by epoch."""
    return [
        {
            "epoch": e.epoch,
            "inner_training_batch_loss": e.training_batch_loss,
            "inner_stop_egfr_mae": e.stopping_metrics.egfr_mae,
            "inner_stop_her2_mae": e.stopping_metrics.her2_mae,
            "inner_stop_balanced_mae": e.stopping_metrics.balanced_mae,
        }
        for e in history
    ]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Cannot write empty result CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(model, *, fold: int, epochs: int, config, bank, path: Path) -> None:
    """Save restorable PyTorch weights with architecture and data provenance."""
    if config.architecture != "gatv2" or not isinstance(model, MolecularGATv2):
        raise TypeError("Cannot save a non-GATv2 model as a GATv2 checkpoint")
    payload = {
        "state_dict": {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()},
        "model_config": {
            "hidden_dim": config.hidden_dim,
            "num_layers": config.num_layers,
            "heads": config.attention_heads,
            "dropout": config.dropout,
            "attention_dropout": config.attention_dropout,
        },
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
        "purpose": "fit_only_gatv2_reference_cv_not_final_model",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if loaded["model_config"] != payload["model_config"]:
        raise ValueError("Saved GATv2 checkpoint configuration differs from model")
    restored = MolecularGATv2(**loaded["model_config"])
    restored.load_state_dict(loaded["state_dict"], strict=True)
    if not all(torch.equal(a, b) for a, b in zip(
        payload["state_dict"].values(), restored.state_dict().values(), strict=True
    )):
        raise ValueError("GATv2 checkpoint weights failed roundtrip verification")


def log_numeric_metrics(mlflow, metrics: dict, prefix: str) -> None:
    mlflow.log_metrics({
        f"{prefix}{key}": float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and math.isfinite(value)
    })


def output_paths(report_dir: Path, model_dir: Path) -> dict[str, Path]:
    return {
        "report": report_dir / f"{PREFIX}_5fold_cv.json",
        "metrics": report_dir / f"{PREFIX}_5fold_metrics.csv",
        "oof": report_dir / f"{PREFIX}_oof_predictions.csv",
        **{
            f"checkpoint_{fold}": model_dir / f"{PREFIX}_fold{fold}.pt"
            for fold in range(5)
        },
        **{
            f"history_{fold}": report_dir / f"{PREFIX}_fold{fold}_epochs.csv"
            for fold in range(5)
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default=Path("data/interim/M2_1_scaffold_split_molecules.csv"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/interim/M2_1_scaffold_split_manifest.csv"))
    parser.add_argument("--summary", type=Path,
                        default=Path("reports/audits/M2_1_scaffold_split_summary.json"))
    parser.add_argument("--gine-oof", type=Path, default=REFERENCE_OOF)
    parser.add_argument("--report-dir", type=Path, default=Path("reports/milestones"))
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--mlflow-uri", default="http://127.0.0.1:5000")
    parser.add_argument("--mlflow-experiment", default=EXPERIMENT)
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow replacement of existing GATv2 output files (never GINE files)")
    args = parser.parse_args(argv)
    if args.max_epochs < 20:
        parser.error("Use at least 20 epochs for GATv2 reference CV; no engineering smoke mode")
    if args.patience < 1:
        parser.error("patience must be >= 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    if not args.mlflow_uri.startswith(("http://", "https://")):
        parser.error("Explicit HTTP(S) MLflow server required (no new local database)")

    paths = output_paths(args.report_dir, args.model_dir)
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing and not args.overwrite:
        parser.error(f"GATv2 results already exist; refusing to overwrite: {existing}")
    config = replace(
        GNNTrainingConfig(),
        architecture="gatv2",
        attention_heads=4,
        attention_dropout=0.0,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )
    source = load_modeling_pipeline(args.dataset, args.manifest, args.summary)
    if source.data.dataset_sha256 != DATASET_HASH:
        raise ValueError("Frozen M2.1 dataset SHA-256 mismatch")
    split = make_scaffold_calibration_split(
        source, seed=719, calibration_fraction=0.20, threshold_fraction=0.20,
        minimum_target_labels=10, minimum_dual_per_class=50, max_attempts=1,
    )
    fit_only = build_fit_only_tuning_pipeline(source, split, n_splits=5)
    if fit_only.stage_c_assignment_sha256 != FROZEN_STAGE_C_ASSIGNMENT_SHA256:
        raise ValueError("Frozen Stage C assignment hash mismatch")
    prior_folds = read_reference_fold_membership(args.gine_oof)
    verify_reference_fold_membership(fit_only, prior_folds)
    bank = build_fit_only_graph_bank(fit_only)
    if bank.molecules != 5364 or len(fit_only.folds) != 5:
        raise ValueError("Unexpected fit-only count or number of CV folds")

    import mlflow

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.mlflow_experiment)
    device = torch.device(args.device)
    fold_reports: list[dict] = []
    all_rows: list[dict] = []
    mlflow_run_id = None
    oof_sha = hashlib.sha256(args.gine_oof.read_bytes()).hexdigest()
    with mlflow.start_run(run_name="M3_6_GATv2_Reference_5Fold_CV") as parent:
        mlflow_run_id = parent.info.run_id
        mlflow.set_tags({
            "stage": "M3.6", "model_family": "GATv2", "architecture": "gatv2",
            "run_type": "reference_cv_after_gine", "cv_type": "fit_only_scaffold_5fold",
            "cv_holdout_scored": "true", "calibration_scored": "false",
            "threshold_scored": "false", "development_validation_scored": "false",
            "locked_test_scored": "false", "graph_schema_version": bank.graph_schema_version,
            "dataset_sha256": bank.dataset_sha256,
            "outer_manifest_sha256": bank.manifest_sha256,
            "stage_c_assignment_sha256": bank.stage_c_assignment_sha256,
            "gine_reference_oof_sha256": oof_sha,
        })
        mlflow.log_params({f"gnn_{key}": value for key, value in asdict(config).items()})
        mlflow.log_param("device", args.device)
        print(f"MLflow GATv2 parent run: {parent.info.run_id}")
        for fold_num in range(5):
            view = select_fold_graphs(fit_only, bank, fold_num)
            print(f"Fold {fold_num}: selecting inner epoch on {len(view.fit)} CV-fit graphs")
            selected = train_with_inner_stopping(view, config=config, device=device)
            model = refit_cv_training_graphs(
                view, epochs=selected.best_epoch, config=config, device=device,
            )
            if not isinstance(model, MolecularGATv2):
                raise TypeError("GATv2 training factory returned a different architecture")
            rows = predict_oof(
                model, view.holdout, fold=fold_num,
                batch_size=config.batch_size, device=device,
            )
            metrics = score_oof_rows(rows)
            fold_reports.append({
                "fold": fold_num,
                "fit_n": len(view.fit),
                "holdout_n": len(view.holdout),
                "inner_fit_n": selected.inner_train_molecules,
                "inner_stop_n": selected.inner_stop_molecules,
                "best_epoch": selected.best_epoch,
                "epochs_ran": selected.epochs_ran,
                **metrics,
            })
            all_rows.extend(rows)
            checkpoint = paths[f"checkpoint_{fold_num}"]
            history = paths[f"history_{fold_num}"]
            save_checkpoint(
                model, fold=fold_num, epochs=selected.best_epoch,
                config=config, bank=bank, path=checkpoint,
            )
            write_csv(history, epoch_rows(selected.history))
            with mlflow.start_run(run_name=f"M3_6_GATv2_reference_fold_{fold_num}", nested=True):
                mlflow.set_tags({"fold": str(fold_num), "role": "cv_holdout_score",
                                 "architecture": "gatv2"})
                mlflow.log_params({
                    "cv_fit_n": len(view.fit), "cv_holdout_n": len(view.holdout),
                    "best_inner_epoch": selected.best_epoch,
                    "epochs_ran": selected.epochs_ran,
                })
                log_numeric_metrics(mlflow, metrics, "cv_")
                for epoch in selected.history:
                    mlflow.log_metric("inner_train_loss", epoch.training_batch_loss,
                                      step=epoch.epoch)
                    mlflow.log_metric("inner_stop_egfr_mae", epoch.stopping_metrics.egfr_mae,
                                      step=epoch.epoch)
                    mlflow.log_metric("inner_stop_her2_mae", epoch.stopping_metrics.her2_mae,
                                      step=epoch.epoch)
                mlflow.log_artifact(str(checkpoint), artifact_path="checkpoints")
                mlflow.log_artifact(str(history), artifact_path="evidence")
            print(
                f"  EGFR MAE={metrics['egfr_mae']:.4f}; "
                f"HER2 MAE={metrics['her2_mae']:.4f}; "
                f"dual AP={metrics['dual_min_prediction_ap']}; "
                f"selected epoch={selected.best_epoch}"
            )

        verify_oof_coverage(all_rows, bank.ordered_ids, expected_folds=5)
        if {row["structure_id"]: row["fold"] for row in all_rows} != prior_folds:
            raise ValueError("Final GATv2 OOF assignment differs from reference GINE OOF")
        pooled = score_oof_rows(all_rows)
        report = {
            "purpose": "GATv2 edge-aware reference CV on fit-only Stage C molecules",
            "architecture": "gatv2",
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
            "gine_reference_oof_sha256": oof_sha,
            "mlflow_parent_run_id": mlflow_run_id,
            "folds": fold_reports,
            "pooled_oof": pooled,
        }
        write_csv(paths["metrics"], fold_reports)
        write_csv(paths["oof"], all_rows)
        paths["report"].parent.mkdir(parents=True, exist_ok=True)
        paths["report"].write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        log_numeric_metrics(mlflow, pooled, "pooled_oof_")
        for key in ("report", "metrics", "oof"):
            mlflow.log_artifact(str(paths[key]), artifact_path="evidence")
        print(
            f"GATv2 pooled OOF: EGFR MAE={pooled['egfr_mae']:.4f}, "
            f"HER2 MAE={pooled['her2_mae']:.4f}, "
            f"dual AP={pooled['dual_min_prediction_ap']}"
        )
        print(f"Saved {paths['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
