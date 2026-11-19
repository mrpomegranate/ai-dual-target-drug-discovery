"""M3.7d: frozen protocol preflight, then explicitly authorized ONE-TIME test scoring.

Preflight never parses, exposes, or scores test potency labels.
Execution is irrevocably marked before those labels are parsed. No training, tuning, conformal fitting or overrides.

Execution intentionally does not permit reruns or output overwrites. If a run
fails after unlocking, preserve the marker and artifacts for an audit; do not
rerun or revise the protocol using any observed test labels.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from scripts.evaluate_rf_gine_dev_validation import (
    EXPECTED_DATASET,
    EXPECTED_MANIFEST,
    EXPECTED_STAGE_C,
    EXPERIMENT,
    MODELS,
    TARGETS,
    make_validation_frame,
    sha256_path,
    verify_frozen_report,
)

UNLOCK_TOKEN = "UNLOCK_M3_7D_TEST_ONCE"
STAGE = "M3.7d"
OUTPUT_PREFIX = "M3_7d_locked_test"


def model_paths_from_dir(model_dir: Path) -> dict[str, Path]:
    return {
        "egfr_rf": model_dir / "M3_7b_final_RF_EGFR.joblib",
        "her2_rf": model_dir / "M3_7b_final_RF_HER2.joblib",
        "gine": model_dir / "M3_7b_final_GINE_fit_only.pt",
    }


def report_paths(report_dir: Path) -> dict[str, Path]:
    root = report_dir / OUTPUT_PREFIX
    return {
        "predictions": root.with_name(f"{OUTPUT_PREFIX}_predictions.csv"),
        "metrics_csv": root.with_name(f"{OUTPUT_PREFIX}_metrics.csv"),
        "metrics_json": root.with_name(f"{OUTPUT_PREFIX}_metrics.json"),
        "scaffolds": root.with_name(f"{OUTPUT_PREFIX}_scaffold_coverage.csv"),
        "potency": root.with_name(f"{OUTPUT_PREFIX}_potency_coverage.csv"),
        "coverage_plot": root.with_name(f"{OUTPUT_PREFIX}_coverage.png"),
        "mae_plot": root.with_name(f"{OUTPUT_PREFIX}_mae.png"),
    }


def marker_path(report_dir: Path) -> Path:
    return report_dir / "M3_7d_LOCKED_TEST_CONSUMED.json"


def frozen_protocol_path(report_dir: Path) -> Path:
    return report_dir / "M3_7d_frozen_test_protocol.json"


def validate_dev_report(dev: dict, cal: dict, cal_sha: str) -> None:
    """Require M3.7c to be completed without unlocking or changing models."""
    if (dev.get("purpose") != "independent_frozen_development_validation_no_test"
            or dev.get("evaluation_role") != "development_validation"
            or dev.get("molecules") != 1087
            or dev.get("locked_test_scored") is not False):
        raise ValueError("Development-validation report invalid or test already scored")
    if (dev.get("models") != list(MODELS) or dev.get("nominal_marginal_coverage") != .90
            or dev.get("calibration_scored") is not False
            or dev.get("threshold_scored") is not False):
        raise ValueError("Development-validation protocol differs from frozen recipe")
    summaries = dev.get("model_metrics", [])
    if (len(summaries) != 6
            or {(r.get("target"), r.get("model")) for r in summaries}
            != {(t, m) for t in TARGETS for m in MODELS}):
        raise ValueError("Incomplete development-validation scores")
    provenance = dev.get("provenance", {})
    for field, expected in (
        ("dataset_sha256", EXPECTED_DATASET),
        ("outer_manifest_sha256", EXPECTED_MANIFEST),
        ("stage_c_assignment_sha256", EXPECTED_STAGE_C),
        ("m3_7b_calibrator_report_sha256", cal_sha),
    ):
        if provenance.get(field) != expected:
            raise ValueError(f"Development-validation provenance mismatch: {field}")
    if provenance.get("model_file_sha256") != cal["provenance"]["model_file_sha256"]:
        raise ValueError("Development-validation used different model checkpoints")
    if provenance.get("m3_7b_mlflow_run_id") != cal.get("mlflow_run_id"):
        raise ValueError("Development-validation used a different calibration run")


def verify_test_role(source, test_records) -> None:
    """Cross-check the unlocked test against train and validation identities/scaffolds."""
    if len(test_records) != 1087:
        raise ValueError("Expected 1,087 locked-test molecules")
    train, validation = source.data.train, source.data.validation
    reserved_ids = {r.structure_id for r in (*train, *validation)}
    reserved_scaffolds = {r.scaffold_group_id for r in (*train, *validation)}
    ids = {r.structure_id for r in test_records}
    scaffolds = {r.scaffold_group_id for r in test_records}
    if (len(ids) != len(test_records) or ids & reserved_ids
            or scaffolds & reserved_scaffolds):
        raise ValueError("Locked test overlaps model-development molecules or scaffolds")


def build_frozen_protocol(
    *,
    dataset: Path,
    manifest: Path,
    summary: Path,
    calibrators: Path,
    dev_metrics: Path,
    model_paths: dict[str, Path],
    minimum_scaffold_n: int,
) -> dict:
    """Construct a deterministic protocol without parsing test potency labels."""
    repo_root = Path(__file__).resolve().parent.parent
    source_root = repo_root / "ml/src/dual_target"
    code_files = {
        "test_driver": Path(__file__),
        "test_loader": source_root / "evaluation/locked_test_io.py",
        "scoring": source_root / "evaluation/final_validation.py",
        "development_driver": repo_root / "scripts/evaluate_rf_gine_dev_validation.py",
        "m2_dataset_loader": source_root / "data/modeling_dataset.py",
        "m2_pipeline": source_root / "data/pipeline.py",
        "graph_features": source_root / "features/molecular_graph.py",
        "graph_loader": source_root / "data/gnn_dataset.py",
        "gine_model": source_root / "models/gnn.py",
        "morgan_features": source_root / "morgan_features.py",
        "stage_c_split": source_root / "calibration/scaffold_calibration.py",
    }
    # Use absolute repo-relative paths for code so calling from project root works.
    code_files = {key: path.resolve() for key, path in code_files.items()}
    for path in (*model_paths.values(), dataset, manifest, summary,
                 calibrators, dev_metrics, *code_files.values()):
        if not path.is_file():
            raise FileNotFoundError(f"Required frozen input unavailable: {path}")
    return {
        "purpose": "one_time_frozen_scaffold_heldout_test_comparison",
        "stage": STAGE,
        "expected_test_molecules": 1087,
        "models": list(MODELS),
        "targets": list(TARGETS),
        "model_configuration": {
            "rf_models": "M3.7b frozen EGFR and HER2 regression checkpoints",
            "gine": {"architecture": "gine", "hidden_dim": 128,
                     "num_layers": 3, "dropout": .1, "refit_epochs": 47},
            "ensemble_rf_weight": .75,
        },
        "evaluation": {
            "pIC50": ["mae", "spearman"],
            "paired_activity": "AP(min(predicted_EGFR_pIC50,predicted_HER2_pIC50)); both measured; both >6",
            "conformal": "frozen individual-target 90% symmetric split-conformal q",
            "joint_coverage": "descriptive only; no 90% joint guarantee",
            "potency_bins": ["<6", "6–<7", "7–<8", "≥8"],
            "minimum_scaffold_n": minimum_scaffold_n,
            "no_model_selection_from_test": True,
        },
        "provenance": {
            "dataset_sha256": sha256_path(dataset),
            "outer_manifest_sha256": sha256_path(manifest),
            "audit_summary_sha256": sha256_path(summary),
            "calibrator_report_sha256": sha256_path(calibrators),
            "dev_validation_metrics_sha256": sha256_path(dev_metrics),
            "model_file_sha256": {key: sha256_path(path) for key, path in model_paths.items()},
            "code_sha256": {key: sha256_path(path) for key, path in code_files.items()},
        },
    }


def require_one_time_authorization(
    *, token: str | None, expected_sha: str | None, protocol_file: Path,
    consumed_marker: Path, outputs: dict[str, Path], current_protocol: dict,
) -> str:
    """Fail closed without opening any test labels or creating the marker."""
    if consumed_marker.exists() or any(path.exists() for path in outputs.values()):
        raise FileExistsError("Test evaluation previously started: rerunning is prohibited")
    if token != UNLOCK_TOKEN:
        raise PermissionError("Explicit --confirm-unlock is required")
    if not protocol_file.is_file():
        raise FileNotFoundError("Frozen protocol missing: run --preflight first")
    actual_sha = sha256_path(protocol_file)
    if not expected_sha or actual_sha != expected_sha.lower():
        raise ValueError("Frozen protocol SHA-256 must match the preflight output")
    if json.loads(protocol_file.read_text(encoding="utf-8")) != current_protocol:
        raise ValueError("Frozen protocol or source/model files changed after preflight")
    return actual_sha


def reserve_one_time_test(consumed_marker: Path, protocol_sha256: str, mlflow_run_id: str) -> None:
    """Atomically mark the test consumed BEFORE the first test label is read."""
    consumed_marker.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "stage": STAGE,
        "status": "test_unlock_started_do_not_rerun",
        "protocol_sha256": protocol_sha256,
        "mlflow_run_id": mlflow_run_id,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "notes": "Marker created before reading test labels. Never delete for score retries.",
    }
    with consumed_marker.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(record, indent=2) + "\n")
        handle.flush()


def make_model_predictions(records, paths: dict[str, Path], device_name: str):
    """Read-only inference, matching the already-validated M3.7c inference recipe."""
    import joblib
    import numpy as np
    import torch
    from dual_target.data.gnn_dataset import make_graph_loader
    from dual_target.features.molecular_graph import (
        GRAPH_SCHEMA_VERSION,
        molecule_to_graph,
    )
    from dual_target.models.gnn import MolecularGNN
    from dual_target.morgan_features import make_morgan_fingerprints

    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    ids = [r.structure_id for r in records]
    features = make_morgan_fingerprints(
        [r.smiles for r in records], radius=2, n_bits=2048, include_chirality=True,
    )
    rf = {}
    for target, key in (("egfr", "egfr_rf"), ("her2", "her2_rf")):
        estimator = joblib.load(paths[key])  # Only local, previously-hashed checkpoints.
        estimates = estimator.predict(features)
        if estimates.shape != (len(records),) or not np.isfinite(estimates).all():
            raise ValueError(f"Invalid {target} RF test predictions")
        rf[target] = dict(zip(ids, map(float, estimates), strict=True))
    payload = torch.load(paths["gine"], map_location="cpu", weights_only=True)
    config = payload.get("model_config", {})
    expected_cfg = {"architecture": "gine", "hidden_dim": 128,
                    "num_layers": 3, "dropout": .1}
    if any(config.get(k) != v for k, v in expected_cfg.items()):
        raise ValueError("Frozen GINE architecture changed")
    if (payload.get("refit_epochs") != 47
            or payload.get("dataset_sha256") != EXPECTED_DATASET
            or payload.get("outer_manifest_sha256") != EXPECTED_MANIFEST
            or payload.get("stage_c_assignment_sha256") != EXPECTED_STAGE_C
            or payload.get("graph_schema_version") != GRAPH_SCHEMA_VERSION):
        raise ValueError("Frozen GINE checkpoint provenance mismatch")
    device = torch.device(device_name)
    model = MolecularGNN(hidden_dim=128, num_layers=3, dropout=.1).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    graphs = tuple(molecule_to_graph(r) for r in records)
    predictions = {"egfr": {}, "her2": {}}
    with torch.inference_mode():
        for cpu_batch in make_graph_loader(graphs, batch_size=64, shuffle=False):
            values = model(cpu_batch.to(device)).detach().cpu().numpy()
            if (values.ndim != 2 or values.shape[1] != 2
                    or not np.isfinite(values).all()):
                raise ValueError("Invalid GINE inference output")
            for sid, pair in zip(cpu_batch.structure_id, values, strict=True):
                if sid in predictions["egfr"]:
                    raise ValueError("Duplicate GINE test prediction")
                for target, value in zip(TARGETS, pair, strict=True):
                    predictions[target][sid] = float(value)
    for target in TARGETS:
        if set(predictions[target]) != set(ids):
            raise ValueError(f"Incomplete {target} GINE test predictions")
    return make_validation_frame(records, rf, predictions)


def evaluate_frozen_test(frame, cal_report: dict, *, minimum_scaffold_n: int):
    """Reuse M3.7c's fixed metrics and descriptive subgroup evaluators."""
    import numpy as np
    import pandas as pd
    from dual_target.evaluation.final_validation import (
        make_detail_rows,
        score_dual_activity,
        score_potency_bins,
        score_scaffolds,
        score_target,
        validate_prediction_table,
    )

    validate_prediction_table(frame, expected_n=1087)
    metrics, detail_frames = [], []
    for target in TARGETS:
        for model in MODELS:
            frozen_q = cal_report["calibrators"][model][target]["absolute_error_quantile"]
            summary, details = score_target(frame, target=target, model=model,
                                            quantile=float(frozen_q))
            metrics.append(summary)
            detail_frames.append(details)
    detail = make_detail_rows(detail_frames)
    scaffold = score_scaffolds(detail, minimum_n=minimum_scaffold_n)
    potency = score_potency_bins(detail)
    dual = score_dual_activity(frame)
    paired = frame.loc[frame.egfr_mask.astype(bool) & frame.her2_mask.astype(bool)]
    joint = {}
    for model in MODELS:
        hit = []
        for target in TARGETS:
            q = cal_report["calibrators"][model][target]["absolute_error_quantile"]
            values = np.abs(paired[f"{target}_{model}_pred"] - paired[f"{target}_pic50"])
            hit.append(values <= q)
        joint[model] = {
            "paired_n": len(paired),
            "observed_simultaneous_interval_coverage": float((hit[0] & hit[1]).mean()),
            "joint_coverage_guaranteed": False,
        }
    return metrics, pd.DataFrame(metrics), scaffold, potency, dual, joint


def write_test_figures(metrics: list[dict], outputs: dict[str, Path]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [(t, m) for t in TARGETS for m in MODELS]
    selected = [next(r for r in metrics if (r["target"], r["model"]) == label)
                for label in labels]
    ticklabels = [f"{t.upper()}\n{m}" for t, m in labels]
    for key, metric, ylabel, title, threshold in (
        ("coverage_plot", "empirical_coverage", "Empirical interval coverage",
         "Locked-test coverage of frozen nominal 90% intervals", .90),
        ("mae_plot", "mae", "MAE (pIC50; lower is better)",
         "Locked-test potency regression error", None),
    ):
        fig, ax = plt.subplots(figsize=(9.2, 4.6))
        ax.bar(range(len(selected)), [r[metric] for r in selected])
        ax.set_xticks(range(len(selected)), ticklabels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if threshold is not None:
            ax.axhline(threshold, linestyle="--", linewidth=1.4, label="Nominal 90%")
            ax.set_ylim(0, 1.05)
            ax.legend()
        fig.tight_layout()
        fig.savefig(outputs[key], dpi=170)
        plt.close(fig)


def run_preflight(args, cal: dict, dev: dict, paths: dict[str, Path]) -> int:
    protocol_path = frozen_protocol_path(args.report_dir)
    consumed = marker_path(args.report_dir)
    if consumed.exists() or any(path.exists() for path in report_paths(args.report_dir).values()):
        raise FileExistsError("Locked test was already scored or unlocked")
    protocol = build_frozen_protocol(
        dataset=args.dataset, manifest=args.manifest, summary=args.summary,
        calibrators=args.calibrators, dev_metrics=args.dev_metrics,
        model_paths=paths, minimum_scaffold_n=args.minimum_scaffold_n,
    )
    if protocol["provenance"]["dataset_sha256"] != EXPECTED_DATASET or (
        protocol["provenance"]["outer_manifest_sha256"] != EXPECTED_MANIFEST
    ):
        raise ValueError("Frozen dataset and scaffold manifest hashes differ")
    # load_modeling_pipeline() verifies all source/manifest identities, but
    # intentionally does not read/return test labels.
    from dual_target.calibration.scaffold_calibration import (
        make_scaffold_calibration_split,
    )
    from dual_target.data.pipeline import load_modeling_pipeline

    source = load_modeling_pipeline(args.dataset, args.manifest, args.summary)
    split = make_scaffold_calibration_split(
        source, seed=719, calibration_fraction=.20, threshold_fraction=.20,
        minimum_target_labels=10, minimum_dual_per_class=50, max_attempts=1,
    )
    if (split.assignment_sha256 != EXPECTED_STAGE_C
            or (len(split.fit), len(split.calibration), len(split.threshold)) != (5364, 1658, 1670)):
        raise ValueError("Stage C assignments have changed")
    if len(source.data.validation) != 1087 or len(source.data.train) != 8692:
        raise ValueError("Frozen M2.1 outer partitions have changed")
    # Exercise real checkpoint deserialization and inference on fit-role
    # examples before the irreversible test unlock. No scores are computed.
    make_model_predictions(tuple(split.fit[:2]), paths, args.device)
    import mlflow

    mlflow.set_tracking_uri(args.mlflow_uri)
    experiment = mlflow.get_experiment_by_name(args.mlflow_experiment)
    if experiment is None or experiment.lifecycle_stage != "active":
        raise ValueError("Existing MLflow experiment unavailable at preflight")
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(protocol, indent=2, allow_nan=False) + "\n"
    if protocol_path.exists():
        if protocol_path.read_text(encoding="utf-8") != content:
            raise FileExistsError("Existing frozen protocol differs: refusing overwrite")
    else:
        with protocol_path.open("x", encoding="utf-8") as stream:
            stream.write(content)
    print("PREFLIGHT PASSED: no locked-test label parsed/scored; checkpoints verified.")
    print("Frozen protocol:", protocol_path)
    print("Protocol SHA-256:", sha256_path(protocol_path))
    print("To score once, use --evaluate --confirm-unlock", UNLOCK_TOKEN,
          "--expected-protocol-sha256 <SHA_ABOVE>")
    return 0


def run_evaluate(args, cal: dict, dev: dict, paths: dict[str, Path]) -> int:
    import mlflow

    protocol = build_frozen_protocol(
        dataset=args.dataset, manifest=args.manifest, summary=args.summary,
        calibrators=args.calibrators, dev_metrics=args.dev_metrics,
        model_paths=paths, minimum_scaffold_n=args.minimum_scaffold_n,
    )
    protocol_sha = require_one_time_authorization(
        token=args.confirm_unlock, expected_sha=args.expected_protocol_sha256,
        protocol_file=frozen_protocol_path(args.report_dir),
        consumed_marker=marker_path(args.report_dir),
        outputs=report_paths(args.report_dir), current_protocol=protocol,
    )
    if (protocol["provenance"]["dataset_sha256"] != EXPECTED_DATASET
            or protocol["provenance"]["outer_manifest_sha256"] != EXPECTED_MANIFEST):
        raise ValueError("M2.1 dataset or scaffold manifest changed after freeze")
    # Safe loader: excludes test labels, checks Stage C and outer hashes.
    from dual_target.calibration.scaffold_calibration import (
        make_scaffold_calibration_split,
    )
    from dual_target.data.pipeline import load_modeling_pipeline
    source = load_modeling_pipeline(args.dataset, args.manifest, args.summary)
    split = make_scaffold_calibration_split(
        source, seed=719, calibration_fraction=.20, threshold_fraction=.20,
        minimum_target_labels=10, minimum_dual_per_class=50, max_attempts=1,
    )
    if split.assignment_sha256 != EXPECTED_STAGE_C:
        raise ValueError("Stage C assignment changed after freeze")
    mlflow.set_tracking_uri(args.mlflow_uri)
    experiment = mlflow.get_experiment_by_name(args.mlflow_experiment)
    if experiment is None or experiment.lifecycle_stage != "active":
        raise ValueError("Existing MLflow experiment not available")
    outputs = report_paths(args.report_dir)
    with mlflow.start_run(experiment_id=experiment.experiment_id,
                          run_name="M3_7d_OneTime_RF_GINE_LockedTest") as run:
        # From this line onward, test labels are considered consumed EVEN if
        # a later inference or artifact logging error interrupts the process.
        reserve_one_time_test(marker_path(args.report_dir), protocol_sha, run.info.run_id)
        from dual_target.evaluation.locked_test_io import load_locked_test_records
        test = load_locked_test_records(
            args.dataset, args.manifest, args.summary,
            authorization_token=args.confirm_unlock,
            expected_dataset_sha256=EXPECTED_DATASET,
            expected_manifest_sha256=EXPECTED_MANIFEST, expected_test_n=1087,
        )
        verify_test_role(source, test)
        frame = make_model_predictions(test, paths, args.device)
        metrics, metrics_df, scaffolds, potency, dual, joint = evaluate_frozen_test(
            frame, cal, minimum_scaffold_n=args.minimum_scaffold_n,
        )
        report = {
            "purpose": "one_time_locked_test_scaffold_heldout_RF_vs_GINE_vs_ensemble",
            "evaluation_role": "locked_test",
            "molecules": 1087,
            "models": list(MODELS),
            "nominal_marginal_coverage": .90,
            "model_metrics": metrics,
            "dual_activity": dual,
            "paired_joint_interval_coverage_descriptive": joint,
            "scaffold_diagnostics": {
                "minimum_n_for_ranking": args.minimum_scaffold_n,
                "note": "Only descriptive; no scaffold-conditional coverage guarantee.",
                "group_counts": {t: int(scaffolds.loc[
                    (scaffolds.target == t) & (scaffolds.model == "rf"),
                    "scaffold_group_id"].nunique()) for t in TARGETS},
                "reportable_counts": {t: int(scaffolds.loc[
                    (scaffolds.target == t) & (scaffolds.model == "rf")
                    & scaffolds.meets_minimum_n].shape[0]) for t in TARGETS},
            },
            "provenance": {
                **protocol["provenance"], "frozen_protocol_sha256": protocol_sha,
                "m3_7b_mlflow_run_id": cal.get("mlflow_run_id"),
                "m3_7c_mlflow_run_id": dev.get("mlflow_run_id"),
            },
            "fit_scored": False,
            "calibration_scored": False,
            "threshold_scored": False,
            "development_validation_scored": False,
            "locked_test_scored": True,
            "mlflow_run_id": run.info.run_id,
            "limitation": "No model, ensemble, thresholds, or intervals updated on test. "
                          "Scaffold-shift/exchangeability caveat remains; previous RF "
                          "model-selection and calibration partition reuse limits "
                          "unconditional conformal coverage guarantees.",
        }
        args.report_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(outputs["predictions"], index=False)
        metrics_df.to_csv(outputs["metrics_csv"], index=False)
        scaffolds.to_csv(outputs["scaffolds"], index=False)
        potency.to_csv(outputs["potency"], index=False)
        write_test_figures(metrics, outputs)
        outputs["metrics_json"].write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
        mlflow.set_tags({
            "stage": STAGE, "evaluation_role": "locked_test",
            "locked_test_scored": "true", "test_protocol_sha256": protocol_sha,
            "dataset_sha256": EXPECTED_DATASET,
            "stage_c_assignment_sha256": EXPECTED_STAGE_C,
        })
        mlflow.log_params({
            "ensemble_rf_weight": .75, "gine_refit_epochs": 47,
            "nominal_coverage": .9,
            "minimum_scaffold_n": args.minimum_scaffold_n,
        })
        for row in metrics:
            prefix = f"{row['model']}_{row['target']}"
            for key in ("mae", "empirical_coverage", "mean_interval_width_pic50"):
                mlflow.log_metric(f"{prefix}_{key}", row[key])
            if row["spearman"] is not None:
                mlflow.log_metric(f"{prefix}_spearman", row["spearman"])
        for model, entry in dual.items():
            mlflow.log_metric(f"{model}_dual_regression_ap",
                              entry["regression_min_pic50_ap"])
        mlflow.log_artifact(str(frozen_protocol_path(args.report_dir)),
                            artifact_path="locked_test_protocol")
        mlflow.log_artifact(str(marker_path(args.report_dir)),
                            artifact_path="locked_test_protocol")
        for output in outputs.values():
            mlflow.log_artifact(str(output), artifact_path="locked_test_evaluation")
        print("M3.7d ONE-TIME LOCKED TEST COMPLETED. MLflow run:", run.info.run_id)
        for target in TARGETS:
            for model in MODELS:
                metric = next(r for r in metrics if r["target"] == target and r["model"] == model)
                print(f"{model:13s} {target.upper():4s}: n={metric['n_labeled']:4d}, "
                      f"MAE={metric['mae']:.4f}, "
                      f"coverage={metric['empirical_coverage']:.1%}, "
                      f"width={metric['mean_interval_width_pic50']:.3f}")
        for model, vals in dual.items():
            print(f"{model:13s} dual AP={vals['regression_min_pic50_ap']:.4f}")
        for key, output in outputs.items():
            print("Saved", key, ":", output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true", help="Freeze protocol without scoring test labels")
    mode.add_argument("--evaluate", action="store_true", help="One-time labeled test evaluation")
    parser.add_argument("--confirm-unlock", default=None)
    parser.add_argument("--expected-protocol-sha256", default=None)
    parser.add_argument("--dataset", type=Path, default=Path("data/interim/M2_1_scaffold_split_molecules.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("data/interim/M2_1_scaffold_split_manifest.csv"))
    parser.add_argument("--summary", type=Path, default=Path("reports/audits/M2_1_scaffold_split_summary.json"))
    parser.add_argument("--calibrators", type=Path, default=Path("reports/milestones/M3_7b_conformal_calibrators.json"))
    parser.add_argument("--dev-metrics", type=Path, default=Path("reports/milestones/M3_7c_dev_validation_metrics.json"))
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/milestones"))
    parser.add_argument("--minimum-scaffold-n", type=int, default=10)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--mlflow-uri", default="http://127.0.0.1:5000")
    parser.add_argument("--mlflow-experiment", default=EXPERIMENT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.minimum_scaffold_n < 1:
        raise ValueError("minimum-scaffold-n must be positive")
    if not args.mlflow_uri.startswith(("http://", "https://")):
        raise ValueError("Use existing HTTP(S) MLflow server")
    paths = model_paths_from_dir(args.model_dir)
    cal = json.loads(args.calibrators.read_text(encoding="utf-8"))
    verify_frozen_report(cal, model_paths=paths)
    dev = json.loads(args.dev_metrics.read_text(encoding="utf-8"))
    validate_dev_report(dev, cal, sha256_path(args.calibrators))
    if args.preflight:
        if args.confirm_unlock or args.expected_protocol_sha256:
            raise ValueError("Preflight must not accept test-unlock arguments")
        return run_preflight(args, cal, dev, paths)
    return run_evaluate(args, cal, dev, paths)


if __name__ == "__main__":
    raise SystemExit(main())
