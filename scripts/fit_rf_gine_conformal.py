"""M3.7b: freeze, refit RF/GINE on Stage-C fit; conformalize on calibration ONLY.

This script never scores the threshold, development-validation or locked-test
partitions. It uses the frozen RF Optuna studies and the median GINE epoch
selected by the already-completed fit-only five-fold reference CV.

Run from repository root. Requires existing M3.7a conformal module and M3.5a
RF comparison driver. All artifacts are separate from CV checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from dataclasses import asdict
from pathlib import Path

EXPECTED_EPOCHS = (42, 45, 47, 48, 76)
EXPECTED_STAGE_C = "2b99214bce37d30e1eed8f6e22984242492560b2999bb174a8d378604de9171c"
EXPECTED_DATASET = "83a64d02e45ccd9a6310947efcd5cdef9bddc9603edcd55e1d207eb1cafc18b8"
EXPERIMENT = "EGFR_HER2_M2_Scaffold_Benchmark"
ENSEMBLE_RF_WEIGHT = 0.75
ALPHA = 0.10


def select_final_epochs(reference_json: Path) -> tuple[int, str]:
    """Freeze 47 epochs using only previous fit-only CV evidence."""
    payload = json.loads(reference_json.read_text(encoding="utf-8"))
    if payload.get("dataset_sha256") != EXPECTED_DATASET:
        raise ValueError("GINE CV reference dataset digest mismatch")
    if payload.get("stage_c_assignment_sha256") != EXPECTED_STAGE_C:
        raise ValueError("GINE CV Stage C assignment digest mismatch")
    if not payload.get("cv_holdout_scored"):
        raise ValueError("GINE reference CV not completed")
    if any(payload.get(key) for key in (
        "calibration_scored", "threshold_scored",
        "development_validation_scored", "locked_test_scored",
    )):
        raise ValueError("Reference CV unexpectedly touched a reserved partition")
    config = payload.get("config", {})
    selected = {"hidden_dim": 128, "num_layers": 3, "dropout": 0.1,
                "batch_size": 64, "learning_rate": 0.001,
                "weight_decay": 0.0001, "max_epochs": 80,
                "patience": 12, "seed": 719, "huber_delta": 1.0,
                "grad_clip_norm": 5.0}
    if any(config.get(key) != value for key, value in selected.items()):
        raise ValueError("GINE reference configuration differs from frozen recipe")
    if config.get("architecture", "gine") != "gine":
        raise ValueError("GINE reference is a different architecture")
    folds = payload.get("folds", [])
    if len(folds) != 5 or {row.get("fold") for row in folds} != set(range(5)):
        raise ValueError("GINE reference must contain five distinct folds")
    epoch_counts = tuple(sorted(row["best_epoch"] for row in folds))
    if epoch_counts != EXPECTED_EPOCHS:
        raise ValueError("GINE reference best epochs differ from prior evidence")
    return int(statistics.median(epoch_counts)), hashlib.sha256(reference_json.read_bytes()).hexdigest()


def make_ensemble(rf: dict[str, float], gine: dict[str, float],
                  expected_ids: set[str], rf_weight: float = ENSEMBLE_RF_WEIGHT
                  ) -> dict[str, float]:
    """Fixed-weight mixture; refuse missing, extra or nonfinite predictions."""
    if not (0 <= rf_weight <= 1 and math.isfinite(rf_weight)):
        raise ValueError("Invalid ensemble weight")
    if set(rf) != expected_ids or set(gine) != expected_ids:
        raise ValueError("Ensemble predictions must cover precisely expected IDs")
    result = {}
    for sid in expected_ids:
        a, b = float(rf[sid]), float(gine[sid])
        if not (math.isfinite(a) and math.isfinite(b)):
            raise ValueError("Nonfinite ensemble input")
        result[sid] = rf_weight * a + (1 - rf_weight) * b
    return result


def refuse_overwrite(paths: list[Path], *, overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Refusing to overwrite existing final artifacts: " + ", ".join(existing))


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/interim/M2_1_scaffold_split_molecules.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("data/interim/M2_1_scaffold_split_manifest.csv"))
    parser.add_argument("--summary", type=Path, default=Path("reports/audits/M2_1_scaffold_split_summary.json"))
    parser.add_argument("--optuna-db", type=Path, default=Path("reports/optuna/rf_fit_only_seed719.db"))
    parser.add_argument("--gine-cv", type=Path, default=Path("reports/milestones/M3_5_GNN_reference_5fold_cv.json"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/milestones"))
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--mlflow-uri", default="http://127.0.0.1:5000")
    parser.add_argument("--mlflow-experiment", default=EXPERIMENT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if not args.mlflow_uri.startswith(("http://", "https://")):
        parser.error("Requires the existing HTTP(S) MLflow server")
    if not args.optuna_db.is_file():
        parser.error(f"Frozen Optuna database missing: {args.optuna_db}")
    if not args.gine_cv.is_file():
        parser.error(f"GINE reference CV JSON missing: {args.gine_cv}")
    epochs, cv_sha = select_final_epochs(args.gine_cv)

    model_paths = {
        "egfr_rf": args.model_dir / "M3_7b_final_RF_EGFR.joblib",
        "her2_rf": args.model_dir / "M3_7b_final_RF_HER2.joblib",
        "gine": args.model_dir / "M3_7b_final_GINE_fit_only.pt",
    }
    prediction_path = args.report_dir / "M3_7b_calibration_predictions.csv"
    report_path = args.report_dir / "M3_7b_conformal_calibrators.json"
    refuse_overwrite([*model_paths.values(), prediction_path, report_path], overwrite=args.overwrite)

    # Heavy dependencies are loaded only when actually executing the training job.
    import joblib
    import mlflow
    import numpy as np
    import optuna
    import torch
    from compare_rf_gine_oof import STUDIES, validate_study
    from dual_target.calibration.conformal_regression import fit_split_conformal
    from dual_target.calibration.scaffold_calibration import (
        make_scaffold_calibration_split,
    )
    from dual_target.data.gnn_dataset import (
        build_fit_only_graph_bank,
        make_graph_loader,
    )
    from dual_target.data.pipeline import (
        build_fit_only_tuning_pipeline,
        load_modeling_pipeline,
    )
    from dual_target.features.molecular_graph import (
        GRAPH_SCHEMA_VERSION,
        molecule_to_graph,
    )
    from dual_target.features.morgan_pipeline import build_morgan_feature_bank
    from dual_target.models.gnn import MolecularGNN, masked_huber_loss
    from dual_target.models.random_forest import TargetExamples, fit_random_forest
    from dual_target.morgan_features import make_morgan_fingerprints

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    source = load_modeling_pipeline(args.dataset, args.manifest, args.summary)
    if source.data.dataset_sha256 != EXPECTED_DATASET:
        raise ValueError("Dataset hash changed since fit-only CV")
    split = make_scaffold_calibration_split(
        source, seed=719, calibration_fraction=.20, threshold_fraction=.20,
        minimum_target_labels=10, minimum_dual_per_class=50, max_attempts=1,
    )
    fit_only = build_fit_only_tuning_pipeline(source, split, n_splits=5)
    if (split.assignment_sha256 != EXPECTED_STAGE_C
            or fit_only.stage_c_assignment_sha256 != EXPECTED_STAGE_C):
        raise ValueError("Stage C role assignments changed")
    if (len(split.fit), len(split.calibration), len(split.threshold)) != (5364, 1658, 1670):
        raise ValueError("Stage C molecule counts changed")
    fit_ids = {r.structure_id for r in split.fit}
    cal_ids = {r.structure_id for r in split.calibration}
    if fit_ids & cal_ids or {r.scaffold_group_id for r in split.fit} & {
        r.scaffold_group_id for r in split.calibration
    }:
        raise ValueError("Fit/calibration overlap")

    # RF fitting on fit-role labels only, with Optuna selections frozen by CV.
    rf_bank = build_morgan_feature_bank(fit_only.data.train)
    rf_models = {}
    rf_configs = {}
    rf_predictions: dict[str, dict[str, float]] = {}
    storage = f"sqlite:///{args.optuna_db.resolve().as_posix()}"
    calibration_features = make_morgan_fingerprints(
        [r.smiles for r in split.calibration], radius=2,
        n_bits=2048, include_chirality=True,
    )
    for target in ("egfr", "her2"):
        study = optuna.load_study(study_name=STUDIES[target], storage=storage)
        config = validate_study(study, target=target, fit_only=fit_only, bank=rf_bank)
        rf_configs[target] = asdict(config)
        selected = tuple(r for r in split.fit if getattr(r, f"{target}_mask") == 1)
        examples = TargetExamples(
            molecule_ids=tuple(r.structure_id for r in selected),
            scaffold_ids=tuple(r.scaffold_group_id for r in selected),
            features=rf_bank.select_features(selected),
            pic50=np.asarray([getattr(r, f"{target}_pic50") for r in selected], dtype=float),
        )
        rf_models[target] = fit_random_forest(examples, config)
        values = rf_models[target].predict(calibration_features)
        if values.shape != (len(split.calibration),) or not np.isfinite(values).all():
            raise ValueError(f"Invalid {target} RF calibration predictions")
        rf_predictions[target] = {
            r.structure_id: float(v) for r, v in zip(split.calibration, values, strict=True)
        }
        print(f"Refit {target.upper()} RF on {len(selected)} fit-role labels")

    # GINE fresh refit: exactly 47 epochs, on all 5,364 fit-role graphs.
    graph_bank = build_fit_only_graph_bank(fit_only)
    graphs = tuple(graph_bank.graphs_by_id[r.structure_id] for r in split.fit)
    if tuple(graph_bank.ordered_ids) != tuple(r.structure_id for r in split.fit):
        raise ValueError("GINE fit graph order differs from frozen fit partition")
    torch.manual_seed(719)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(719)
    device = torch.device(args.device)
    model = MolecularGNN(hidden_dim=128, num_layers=3, dropout=.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    train_loader = make_graph_loader(graphs, batch_size=64, shuffle=True, seed=719)
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for cpu_batch in train_loader:
            batch = cpu_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            preds = model(batch)
            loss = masked_huber_loss(preds, batch.y, batch.y_mask, delta=1.0)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite GINE training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(loss.detach().item())
        if epoch in (1, 10, 20, 30, 40, epochs):
            print(f"GINE fresh refit epoch {epoch}/{epochs}, fit-role loss={np.mean(losses):.4f}")
    model.eval()
    calibration_graphs = tuple(molecule_to_graph(r) for r in split.calibration)
    gine_predictions: dict[str, dict[str, float]] = {"egfr": {}, "her2": {}}
    with torch.inference_mode():
        for cpu_batch in make_graph_loader(calibration_graphs, batch_size=64, shuffle=False):
            predicted = model(cpu_batch.to(device)).detach().cpu().numpy()
            if predicted.ndim != 2 or predicted.shape[1] != 2 or not np.isfinite(predicted).all():
                raise ValueError("Invalid GINE calibration predictions")
            for sid, values in zip(cpu_batch.structure_id, predicted, strict=True):
                if sid in gine_predictions["egfr"]:
                    raise ValueError("Duplicate GINE calibration prediction")
                gine_predictions["egfr"][sid] = float(values[0])
                gine_predictions["her2"][sid] = float(values[1])
    if any(set(gine_predictions[t]) != cal_ids for t in ("egfr", "her2")):
        raise ValueError("GINE calibration predictions missing molecule IDs")

    ensemble_predictions = {
        target: make_ensemble(rf_predictions[target], gine_predictions[target], cal_ids)
        for target in ("egfr", "her2")
    }
    all_predictions = {
        "rf": rf_predictions,
        "gine": gine_predictions,
        "rf75_gine25": ensemble_predictions,
    }
    calibrators = {}
    for name, target_dict in all_predictions.items():
        calibrators[name] = {}
        for target in ("egfr", "her2"):
            calibrator = fit_split_conformal(
                split, target_dict[target], target=target, model_name=name,
                alpha=ALPHA, expected_assignment_sha256=EXPECTED_STAGE_C,
            )
            calibrators[name][target] = asdict(calibrator)
            print(f"{name:13} {target.upper()}: n={calibrator.n_labels}, "
                  f"90% q={calibrator.absolute_error_quantile:.4f} pIC50, "
                  f"width={2*calibrator.absolute_error_quantile:.4f}")

    rows = []
    for record in split.calibration:
        row = {"structure_id": record.structure_id,
               "scaffold_group_id": record.scaffold_group_id,
               "role": "calibration", "egfr_mask": record.egfr_mask,
               "her2_mask": record.her2_mask}
        for target in ("egfr", "her2"):
            for method in ("rf", "gine", "rf75_gine25"):
                row[f"{target}_{method}_pred"] = all_predictions[method][target][record.structure_id]
        rows.append(row)

    args.model_dir.mkdir(parents=True, exist_ok=True)
    for target in ("egfr", "her2"):
        path = model_paths[f"{target}_rf"]
        joblib.dump(rf_models[target], path)
        restored = joblib.load(path)
        restored_pred = restored.predict(calibration_features)
        if not np.allclose(restored_pred,
                           [rf_predictions[target][r.structure_id] for r in split.calibration],
                           atol=1e-10, rtol=0):
            raise ValueError(f"Restored {target} RF checkpoint changed predictions")

    model_payload = {
        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "model_config": {"hidden_dim": 128, "num_layers": 3, "dropout": .1,
                         "architecture": "gine"},
        "refit_epochs": epochs,
        "selected_from": "median of previous five scaffold-CV inner-stop epochs",
        "purpose": "fit_only_final_candidate_for_calibration_not_test_evaluated",
        "graph_schema_version": GRAPH_SCHEMA_VERSION,
        "dataset_sha256": source.data.dataset_sha256,
        "outer_manifest_sha256": source.data.manifest_sha256,
        "stage_c_assignment_sha256": split.assignment_sha256,
    }
    torch.save(model_payload, model_paths["gine"])
    restored_payload = torch.load(model_paths["gine"], map_location="cpu", weights_only=True)
    restored_model = MolecularGNN(hidden_dim=128, num_layers=3, dropout=.1).to(device)
    restored_model.load_state_dict(restored_payload["state_dict"], strict=True)
    restored_model.eval()
    with torch.inference_mode():
        first_batch = next(iter(make_graph_loader(calibration_graphs, batch_size=64)))
        original = model(first_batch.to(device)).detach().cpu()
        reloaded = restored_model(first_batch.to(device)).detach().cpu()
        if not torch.allclose(original, reloaded, atol=1e-6, rtol=1e-6):
            raise ValueError("Restored GINE checkpoint predictions differ")

    hashes = {}
    for name, path in model_paths.items():
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = {
        "purpose": "fit_role_refit_and_frozen_stage_c_conformal_calibration",
        "n_fit": len(split.fit), "n_calibration": len(split.calibration),
        "n_threshold_unscored": len(split.threshold),
        "ensemble_rf_weight": ENSEMBLE_RF_WEIGHT,
        "alpha": ALPHA,
        "training": {"gine_epochs": epochs, "gine_reference_cv_sha256": cv_sha,
                     "rf_optuna_configs": rf_configs},
        "provenance": {
            "dataset_sha256": source.data.dataset_sha256,
            "outer_manifest_sha256": source.data.manifest_sha256,
            "stage_c_assignment_sha256": split.assignment_sha256,
            "graph_schema_version": GRAPH_SCHEMA_VERSION,
            "model_file_sha256": hashes,
        },
        "calibrators": calibrators,
        "fit_scored": False, "calibration_residuals_used": True,
        "threshold_scored": False, "development_validation_scored": False,
        "locked_test_scored": False,
        "caveat": "Stage C calibration groups participated in earlier full-train Optuna model selection. "
                  "Nominal exchangeability coverage is not guaranteed under scaffold shift.",
    }
    save_csv(prediction_path, rows)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.mlflow_experiment)
    with mlflow.start_run(run_name="M3_7b_RF_GINE_FinalRefit_Conformal90") as run:
        report["mlflow_run_id"] = run.info.run_id
        mlflow.set_tags({
            "stage": "M3.7b", "evaluation_role": "calibration_only",
            "dataset_sha256": source.data.dataset_sha256,
            "stage_c_assignment_sha256": split.assignment_sha256,
            "threshold_scored": "false", "development_validation_scored": "false",
            "locked_test_scored": "false", "gine_architecture": "gine",
        })
        mlflow.log_params({"gine_refit_epochs": epochs, "gine_hidden_dim": 128,
                           "ensemble_rf_weight": ENSEMBLE_RF_WEIGHT,
                           "conformal_alpha": ALPHA})
        for name, group in calibrators.items():
            for target, item in group.items():
                mlflow.log_metric(f"{name}_{target}_calibration_q", item["absolute_error_quantile"])
                mlflow.log_metric(f"{name}_{target}_calibration_n", item["n_labels"])
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for path in (prediction_path, report_path):
            mlflow.log_artifact(str(path), artifact_path="calibration_evidence")
        for path in model_paths.values():
            mlflow.log_artifact(str(path), artifact_path="candidate_checkpoints")
        print(f"MLflow run: {run.info.run_id}")
    print(f"Saved: {report_path}")
    print(f"Saved: {prediction_path}")
    for path in model_paths.values():
        print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
