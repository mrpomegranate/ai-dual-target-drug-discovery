"""M3.7c: read-only, scaffold-held-out development-validation evaluation.

Loads M3.7b FINAL candidate checkpoints and six FROZEN conformal calibrators.
Never fits/recalibrates anything, scores threshold labels, or loads locked-test labels.
Results are model-development evidence: do not retune based on these and then
claim independent performance on the same development-validation partition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

EXPECTED_DATASET = "83a64d02e45ccd9a6310947efcd5cdef9bddc9603edcd55e1d207eb1cafc18b8"
EXPECTED_MANIFEST = "05968ca1ed2a16ca8165e8a28ececa0f361fc9e61a64ac2e69abb1e0e4b7423a"
EXPECTED_STAGE_C = "2b99214bce37d30e1eed8f6e22984242492560b2999bb174a8d378604de9171c"
EXPERIMENT = "EGFR_HER2_M2_Scaffold_Benchmark"
MODELS = ("rf", "gine", "rf75_gine25")
TARGETS = ("egfr", "her2")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_report(report: dict, *, model_paths: dict[str, Path]) -> None:
    """Reject mismatched data, training, mixture, calibrators and model artifacts."""
    if report.get("purpose") != "fit_role_refit_and_frozen_stage_c_conformal_calibration":
        raise ValueError("Not an M3.7b final refit/calibration report")
    if (report.get("n_fit"), report.get("n_calibration"), report.get("n_threshold_unscored")) != (
        5364, 1658, 1670,
    ):
        raise ValueError("Stage C population counts changed")
    if (report.get("ensemble_rf_weight") != .75
            or report.get("alpha") != .1
            or report.get("training", {}).get("gine_epochs") != 47):
        raise ValueError("Final recipe changed after calibration")
    if not report.get("calibration_residuals_used"):
        raise ValueError("Conformal residuals have not been fitted")
    for name in ("threshold_scored", "development_validation_scored", "locked_test_scored"):
        if report.get(name) is not False:
            raise ValueError(f"Previous final refit unexpectedly scored {name}")
    provenance = report.get("provenance", {})
    for key, value in (("dataset_sha256", EXPECTED_DATASET),
                       ("outer_manifest_sha256", EXPECTED_MANIFEST),
                       ("stage_c_assignment_sha256", EXPECTED_STAGE_C),
                       ("graph_schema_version", "m3_molecular_graph_v1")):
        if provenance.get(key) != value:
            raise ValueError(f"Frozen {key} mismatch")
    hashes = provenance.get("model_file_sha256", {})
    if set(hashes) != set(model_paths):
        raise ValueError("Model file hash manifest missing or unexpected keys")
    for key, path in model_paths.items():
        if not path.is_file() or sha256_path(path) != hashes[key]:
            raise ValueError(f"Checkpoint missing or modified: {key}")
    for name in MODELS:
        for target in TARGETS:
            item = report.get("calibrators", {}).get(name, {}).get(target)
            if not isinstance(item, dict):
                raise TypeError(f"Missing {name} {target} calibrator")
            if (item.get("model_name") != name or item.get("target") != target
                    or item.get("alpha") != .1
                    or item.get("stage_c_assignment_sha256") != EXPECTED_STAGE_C
                    or item.get("dataset_sha256") != EXPECTED_DATASET
                    or item.get("outer_manifest_sha256") != EXPECTED_MANIFEST):
                raise ValueError(f"Calibrator provenance mismatch: {name} {target}")
            q = item.get("absolute_error_quantile")
            if not isinstance(q, (int, float)) or not math.isfinite(q) or q < 0:
                raise ValueError(f"Invalid conformal quantile: {name} {target}")
            if item.get("n_labels", 0) < 10 or item.get("n_scaffolds", 0) < 1:
                raise ValueError("Not enough calibrator evidence")


def verify_validation_role(train, validation, split) -> None:
    """Fail closed on wrong evaluation partition and overlapping identities/scaffolds."""
    if len(validation) != 1087:
        raise ValueError("Expected exactly 1,087 frozen development-validation molecules")
    if len({r.structure_id for r in validation}) != len(validation):
        raise ValueError("Duplicate development-validation molecule")
    fit_ids = {r.structure_id for r in train}
    fit_scaffolds = {r.scaffold_group_id for r in train}
    if ({r.structure_id for r in validation} & fit_ids
            or {r.scaffold_group_id for r in validation} & fit_scaffolds):
        raise ValueError("Development-validation overlaps frozen training molecules/scaffolds")
    expected = {r.structure_id for r in (*split.fit, *split.calibration, *split.threshold)}
    if expected != fit_ids:
        raise ValueError("Stage C roles do not reconstruct frozen training")


def make_validation_frame(records, rf_predictions, gine_predictions):
    """Build predictions in the exact frozen validation row order."""
    import pandas as pd

    ids = [record.structure_id for record in records]
    expected = set(ids)
    if len(expected) != len(ids):
        raise ValueError("Duplicate frozen validation molecule")
    for target in TARGETS:
        if set(rf_predictions[target]) != expected or set(gine_predictions[target]) != expected:
            raise ValueError(f"Missing or extra {target} predictions")
    rows = []
    for record in records:
        row = {
            "structure_id": record.structure_id,
            "scaffold_group_id": record.scaffold_group_id,
            "egfr_mask": record.egfr_mask, "her2_mask": record.her2_mask,
            "egfr_pic50": record.egfr_pic50, "her2_pic50": record.her2_pic50,
        }
        for target in TARGETS:
            rf = float(rf_predictions[target][record.structure_id])
            gine = float(gine_predictions[target][record.structure_id])
            if not (math.isfinite(rf) and math.isfinite(gine)):
                raise ValueError(f"Nonfinite {target} prediction")
            row[f"{target}_rf_pred"] = rf
            row[f"{target}_gine_pred"] = gine
            row[f"{target}_rf75_gine25_pred"] = .75 * rf + .25 * gine
        rows.append(row)
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/interim/M2_1_scaffold_split_molecules.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("data/interim/M2_1_scaffold_split_manifest.csv"))
    parser.add_argument("--summary", type=Path, default=Path("reports/audits/M2_1_scaffold_split_summary.json"))
    parser.add_argument("--calibrators", type=Path, default=Path("reports/milestones/M3_7b_conformal_calibrators.json"))
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/milestones"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--minimum-scaffold-n", type=int, default=10)
    parser.add_argument("--mlflow-uri", default="http://127.0.0.1:5000")
    parser.add_argument("--mlflow-experiment", default=EXPERIMENT)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.minimum_scaffold_n < 1:
        raise ValueError("minimum-scaffold-n must be positive")
    if not args.mlflow_uri.startswith(("http://", "https://")):
        raise ValueError("Use the existing HTTP(S) MLflow server")

    path_prefix = args.report_dir / "M3_7c_dev_validation"
    paths = {
        "predictions": path_prefix.with_name(path_prefix.name + "_predictions.csv"),
        "metrics_csv": path_prefix.with_name(path_prefix.name + "_metrics.csv"),
        "metrics_json": path_prefix.with_name(path_prefix.name + "_metrics.json"),
        "scaffolds": path_prefix.with_name(path_prefix.name + "_scaffold_coverage.csv"),
        "potency": path_prefix.with_name(path_prefix.name + "_potency_coverage.csv"),
        "plot": path_prefix.with_name(path_prefix.name + "_coverage.png"),
    }
    existing = [str(p) for p in paths.values() if p.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Refusing to overwrite dev-validation results: " + ", ".join(existing))
    if not args.calibrators.is_file():
        raise FileNotFoundError(args.calibrators)
    model_paths = {
        "egfr_rf": args.model_dir / "M3_7b_final_RF_EGFR.joblib",
        "her2_rf": args.model_dir / "M3_7b_final_RF_HER2.joblib",
        "gine": args.model_dir / "M3_7b_final_GINE_fit_only.pt",
    }
    calibration_report = json.loads(args.calibrators.read_text(encoding="utf-8"))
    verify_frozen_report(calibration_report, model_paths=model_paths)

    # No load_test or test loader exists in this script: only frozen validation.
    import joblib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mlflow
    import numpy as np
    import pandas as pd
    import torch
    from dual_target.calibration.conformal_regression import (
        SplitConformalCalibrator,
        summarize_interval_coverage,
    )
    from dual_target.calibration.scaffold_calibration import (
        make_scaffold_calibration_split,
    )
    from dual_target.data.gnn_dataset import make_graph_loader
    from dual_target.data.pipeline import load_modeling_pipeline
    from dual_target.evaluation.final_validation import (
        make_detail_rows,
        score_dual_activity,
        score_potency_bins,
        score_scaffolds,
        score_target,
        validate_prediction_table,
    )
    from dual_target.features.molecular_graph import (
        GRAPH_SCHEMA_VERSION,
        molecule_to_graph,
    )
    from dual_target.models.gnn import MolecularGNN
    from dual_target.morgan_features import make_morgan_fingerprints

    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    source = load_modeling_pipeline(args.dataset, args.manifest, args.summary)
    if (source.data.dataset_sha256 != EXPECTED_DATASET
            or source.data.manifest_sha256 != EXPECTED_MANIFEST):
        raise ValueError("Dataset or outer manifest has changed")
    split = make_scaffold_calibration_split(
        source, seed=719, calibration_fraction=.20, threshold_fraction=.20,
        minimum_target_labels=10, minimum_dual_per_class=50, max_attempts=1,
    )
    if split.assignment_sha256 != EXPECTED_STAGE_C:
        raise ValueError("Stage C assignment changed")
    validation = tuple(source.data.validation)
    verify_validation_role(source.data.train, validation, split)
    if GRAPH_SCHEMA_VERSION != "m3_molecular_graph_v1":
        raise ValueError("Unexpected molecular graph feature schema")

    ids = [record.structure_id for record in validation]
    rf_predictions = {}
    features = make_morgan_fingerprints(
        [r.smiles for r in validation], radius=2, n_bits=2048, include_chirality=True,
    )
    for target, key in (("egfr", "egfr_rf"), ("her2", "her2_rf")):
        model = joblib.load(model_paths[key])
        values = model.predict(features)
        if values.shape != (len(validation),) or not np.isfinite(values).all():
            raise ValueError(f"Unexpected {target} RF validation prediction")
        rf_predictions[target] = dict(zip(ids, (float(x) for x in values), strict=True))

    payload = torch.load(model_paths["gine"], map_location="cpu", weights_only=True)
    expected_keys = {
        "architecture": "gine", "hidden_dim": 128, "num_layers": 3, "dropout": .1,
    }
    if any(payload.get("model_config", {}).get(k) != v for k, v in expected_keys.items()):
        raise ValueError("GINE checkpoint architecture or configuration mismatch")
    if (payload.get("refit_epochs") != 47
            or payload.get("dataset_sha256") != EXPECTED_DATASET
            or payload.get("outer_manifest_sha256") != EXPECTED_MANIFEST
            or payload.get("stage_c_assignment_sha256") != EXPECTED_STAGE_C
            or payload.get("graph_schema_version") != GRAPH_SCHEMA_VERSION):
        raise ValueError("GINE checkpoint provenance mismatch")
    device = torch.device(args.device)
    gine = MolecularGNN(hidden_dim=128, num_layers=3, dropout=.1).to(device)
    gine.load_state_dict(payload["state_dict"], strict=True)
    gine.eval()
    graphs = tuple(molecule_to_graph(r) for r in validation)
    gine_predictions: dict[str, dict[str, float]] = {"egfr": {}, "her2": {}}
    with torch.inference_mode():
        for batch_cpu in make_graph_loader(graphs, batch_size=64, shuffle=False):
            outputs = gine(batch_cpu.to(device)).detach().cpu().numpy()
            if outputs.ndim != 2 or outputs.shape[1] != 2 or not np.isfinite(outputs).all():
                raise ValueError("Nonfinite or malformed GINE outputs")
            for sid, values in zip(batch_cpu.structure_id, outputs, strict=True):
                if sid in gine_predictions["egfr"]:
                    raise ValueError("Duplicate GINE prediction")
                for target, value in zip(TARGETS, values, strict=True):
                    gine_predictions[target][sid] = float(value)
    frame = make_validation_frame(validation, rf_predictions, gine_predictions)
    validate_prediction_table(frame, expected_n=1087)

    summaries = []
    detail_frames = []
    for target in TARGETS:
        for model in MODELS:
            item = calibration_report["calibrators"][model][target]
            calibrator = SplitConformalCalibrator(**item)
            pred_map = {sid: float(value) for sid, value in zip(
                frame.structure_id, frame[f"{target}_{model}_pred"], strict=True,
            )}
            summary, details = score_target(
                frame, target=target, model=model,
                quantile=calibrator.absolute_error_quantile,
            )
            checked, _ = summarize_interval_coverage(
                calibrator, split, validation, pred_map,
                minimum_scaffold_n=args.minimum_scaffold_n,
            )
            if (summary["n_labeled"] != checked["n_labeled"]
                    or not math.isclose(summary["empirical_coverage"], checked["empirical_coverage"], abs_tol=1e-12)
                    or not math.isclose(summary["mean_interval_width_pic50"], checked["mean_interval_width_pic50"], abs_tol=1e-12)):
                raise ValueError("Independent conformal coverage scorers disagree")
            summaries.append(summary)
            detail_frames.append(details)
            print(f"{model:13s} {target.upper():4s}: n={summary['n_labeled']:4d}, "
                  f"MAE={summary['mae']:.4f}, coverage={summary['empirical_coverage']:.1%}, "
                  f"width={summary['mean_interval_width_pic50']:.3f}")
    details = make_detail_rows(detail_frames)
    scaffold_rows = score_scaffolds(details, minimum_n=args.minimum_scaffold_n)
    potency_rows = score_potency_bins(details)
    dual = score_dual_activity(frame)

    paired = frame.loc[frame.egfr_mask.astype(bool) & frame.her2_mask.astype(bool)]
    joint = {}
    q_by_model = {
        m: {t: calibration_report["calibrators"][m][t]["absolute_error_quantile"]
            for t in TARGETS}
        for m in MODELS
    }
    for model in MODELS:
        hits = []
        for target in TARGETS:
            hits.append(np.abs(paired[f"{target}_{model}_pred"] - paired[f"{target}_pic50"])
                        <= q_by_model[model][target])
        joint[model] = {
            "paired_n": len(paired),
            "observed_simultaneous_interval_coverage": float((hits[0] & hits[1]).mean()),
            "joint_coverage_guaranteed": False,
        }

    report = {
        "purpose": "independent_frozen_development_validation_no_test",
        "evaluation_role": "development_validation",
        "molecules": len(validation),
        "models": list(MODELS),
        "nominal_marginal_coverage": .90,
        "model_metrics": summaries,
        "dual_activity": dual,
        "paired_joint_interval_coverage_descriptive": joint,
        "scaffold_diagnostics": {
            "minimum_n_for_ranking": args.minimum_scaffold_n,
            "note": "Subgroup coverage is descriptive, not guaranteed by marginal conformal calibration.",
            "group_counts": {t: int(scaffold_rows.loc[
                (scaffold_rows.target == t) & (scaffold_rows.model == "rf"),
                "scaffold_group_id"].nunique()) for t in TARGETS},
            "reportable_counts": {t: int(scaffold_rows.loc[
                (scaffold_rows.target == t) & (scaffold_rows.model == "rf")
                & scaffold_rows.meets_minimum_n].shape[0]) for t in TARGETS},
        },
        "provenance": {
            "dataset_sha256": EXPECTED_DATASET,
            "outer_manifest_sha256": EXPECTED_MANIFEST,
            "stage_c_assignment_sha256": EXPECTED_STAGE_C,
            "m3_7b_calibrator_report_sha256": sha256_path(args.calibrators),
            "model_file_sha256": calibration_report["provenance"]["model_file_sha256"],
            "m3_7b_mlflow_run_id": calibration_report.get("mlflow_run_id"),
        },
        "fit_scored": False, "calibration_scored": False, "threshold_scored": False,
        "development_validation_scored": True, "locked_test_scored": False,
        "caveat": "Previously adapted RF hyperparameters used original training CV; no conditional "
                  "or arbitrary scaffold-shift interval coverage guarantee. Do not adjust "
                  "models or calibrators using this development-validation report.",
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(paths["predictions"], index=False)
    pd.DataFrame(summaries).to_csv(paths["metrics_csv"], index=False)
    scaffold_rows.to_csv(paths["scaffolds"], index=False)
    potency_rows.to_csv(paths["potency"], index=False)
    fig, ax = plt.subplots(figsize=(9, 4.7))
    names = [(t, m) for t in TARGETS for m in MODELS]
    values = [next(r["empirical_coverage"] for r in summaries
                   if (r["target"], r["model"]) == (target, model))
              for target, model in names]
    ax.bar(range(len(names)), values)
    ax.axhline(.90, linestyle="--", linewidth=1.4, label="Nominal 90%")
    ax.set_xticks(range(len(names)), [f"{target.upper()}\n{model}" for target, model in names])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Development-validation empirical interval coverage")
    ax.set_title("Frozen conformal intervals on unseen validation scaffolds")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(paths["plot"], dpi=160)
    plt.close(fig)

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.mlflow_experiment)
    with mlflow.start_run(run_name="M3_7c_RF_GINE_DevValidation_Coverage") as run:
        report["mlflow_run_id"] = run.info.run_id
        mlflow.set_tags({
            "stage": "M3.7c", "evaluation_role": "development_validation",
            "dataset_sha256": EXPECTED_DATASET,
            "stage_c_assignment_sha256": EXPECTED_STAGE_C,
            "locked_test_scored": "false", "threshold_scored": "false",
        })
        mlflow.log_params({
            "ensemble_rf_weight": .75, "nominal_coverage": .90,
            "minimum_scaffold_n": args.minimum_scaffold_n,
            "gine_refit_epochs": 47,
        })
        for summary in summaries:
            prefix = f"{summary['model']}_{summary['target']}"
            for key in ("mae", "empirical_coverage", "mean_interval_width_pic50"):
                mlflow.log_metric(f"{prefix}_{key}", summary[key])
        for model, vals in dual.items():
            mlflow.log_metric(f"{model}_dual_regression_ap", vals["regression_min_pic50_ap"])
        paths["metrics_json"].write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
        for path in paths.values():
            mlflow.log_artifact(str(path), artifact_path="development_validation")
        print("MLflow run:", run.info.run_id)
    for name, path in paths.items():
        print(f"Saved {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
