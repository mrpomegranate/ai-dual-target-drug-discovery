"""Compare frozen fit-only Morgan RF regressors against GINE 5-fold OOF.

Use the *existing* Stage C split, five scaffold folds, and corrected RF Optuna
studies. Never load calibration, threshold, development-validation, or test
labels into model fitting. No new tuning or split construction is performed.

Run from the repository root; all defaults are repository-relative.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import optuna
from dual_target.calibration.scaffold_calibration import make_scaffold_calibration_split
from dual_target.data.pipeline import (
    FROZEN_STAGE_C_ASSIGNMENT_SHA256,
    build_fit_only_tuning_pipeline,
    load_modeling_pipeline,
)
from dual_target.evaluation.matched_oof import (
    align_rf_gnn_oof,
    compare_aligned,
    read_gnn_oof_csv,
    write_aligned_csv,
)
from dual_target.evaluation.scaffold_cv import fold_manifest
from dual_target.features.morgan_pipeline import build_morgan_feature_bank
from dual_target.models.random_forest import RFConfig
from dual_target.training.rf_cv import run_rf_scaffold_cv

DATASET_SHA256 = "83a64d02e45ccd9a6310947efcd5cdef9bddc9603edcd55e1d207eb1cafc18b8"
EXPERIMENT = "EGFR_HER2_M2_Scaffold_Benchmark"
STUDIES = {
    "egfr": "M2_2_fit_only_reffirst_200trees_egfr_200trees_5folds",
    "her2": "M2_2_fit_only_reffirst_200trees_her2_200trees_5folds",
}
EXPECTED_BEST = {
    "egfr": {"number": 13, "max_depth": None, "min_samples_leaf": 2,
             "min_samples_split": 3, "max_features": 0.5},
    "her2": {"number": 0, "max_depth": None, "min_samples_leaf": 1,
             "min_samples_split": 2, "max_features": "sqrt"},
}


def compute_fold_manifest_sha256(fit_only) -> str:
    """Match the stable manifest hash used by the existing Optuna module."""
    from dataclasses import asdict as dataclass_to_dict

    rows = [dataclass_to_dict(item) for item in fold_manifest(fit_only.folds)]
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_study(study, *, target: str, fit_only, bank) -> RFConfig:
    """Refuse a changed study, missing provenance, or a different best trial."""
    if target not in STUDIES:
        raise ValueError(f"Unknown target: {target}")
    attrs = study.user_attrs
    required = {
        "task": target,
        "metric_name": "mae",
        "dataset_sha256": fit_only.data.dataset_sha256,
        "manifest_sha256": fit_only.data.manifest_sha256,
        "feature_sha256": bank.fingerprint_sha256,
        "fold_assignments_sha256": compute_fold_manifest_sha256(fit_only),
        "evaluation": "training_only_scaffold_group_cv",
        "reference_first": True,
    }
    for key, value in required.items():
        if attrs.get(key) != value:
            raise ValueError(f"Incompatible {target} Optuna study provenance: {key}")

    base = attrs.get("base_config")
    if not isinstance(base, dict):
        raise TypeError(
            f"{target} Optuna base RF configuration must be a dict, "
            f"got {type(base).__name__}"
        )
    expected_base = {
        "radius": 2, "n_bits": 2048, "include_chirality": True,
        "n_estimators": 200, "seed": 714,
    }
    for key, value in expected_base.items():
        if base.get(key) != value:
            raise ValueError(f"Unexpected {target} RF base configuration: {key}")

    trial = study.best_trial
    expected = EXPECTED_BEST[target]
    if trial.number != expected["number"]:
        raise ValueError(
            f"{target} best trial changed: {trial.number}, expected {expected['number']}"
        )
    parameters = {key: val for key, val in expected.items() if key != "number"}
    if trial.params != parameters:
        raise ValueError(f"{target} selected RF parameters differ from frozen benchmark")
    if not math.isfinite(float(trial.value)):
        raise ValueError(f"{target} Optuna best score is not finite")
    return RFConfig(
        radius=base["radius"],
        n_bits=base["n_bits"],
        include_chirality=base["include_chirality"],
        n_estimators=base["n_estimators"],
        max_depth=parameters["max_depth"],
        min_samples_leaf=parameters["min_samples_leaf"],
        min_samples_split=parameters["min_samples_split"],
        max_features=parameters["max_features"],
        seed=base["seed"],
        n_jobs=int(base.get("n_jobs", -1)),
    )


def build_scaffold_report(matched: list[dict]) -> list[dict]:
    """Descriptive model errors by scaffold, with no extra model selection."""
    groups: dict[str, list[dict]] = {}
    for row in matched:
        groups.setdefault(row["scaffold_group_id"], []).append(row)
    report = []
    for scaffold_id, rows in sorted(groups.items()):
        summary = {"scaffold_group_id": scaffold_id, "n_molecules": len(rows)}
        for target in ("egfr", "her2"):
            subset = [row for row in rows if row[f"{target}_mask"]]
            summary[f"{target}_n"] = len(subset)
            for method in ("rf", "gnn"):
                summary[f"{target}_{method}_mae"] = (
                    sum(abs(row[f"{target}_{method}_pred"] - row[f"{target}_pic50"])
                        for row in subset) / len(subset)
                    if subset else None
                )
        report.append(summary)
    return report


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Cannot write empty report")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def numeric_metrics(metrics: dict, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}{key}": float(value)
        for key, value in metrics.items()
        if type(value) in (int, float) and math.isfinite(float(value))
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default=Path("data/interim/M2_1_scaffold_split_molecules.csv"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/interim/M2_1_scaffold_split_manifest.csv"))
    parser.add_argument("--summary", type=Path,
                        default=Path("reports/audits/M2_1_scaffold_split_summary.json"))
    parser.add_argument("--optuna-db", type=Path,
                        default=Path("reports/optuna/rf_fit_only_seed719.db"))
    parser.add_argument("--gnn-oof", type=Path,
                        default=Path("reports/milestones/M3_5_GNN_reference_oof_predictions.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/milestones"))
    parser.add_argument("--mlflow-uri", default="http://127.0.0.1:5000")
    parser.add_argument("--mlflow-experiment", default=EXPERIMENT)
    args = parser.parse_args(argv)

    if not args.optuna_db.is_file():
        parser.error(f"Optuna database not found: {args.optuna_db}")
    if not args.gnn_oof.is_file():
        parser.error(f"GNN OOF CSV not found: {args.gnn_oof}")
    if not args.mlflow_uri.startswith(("http://", "https://")):
        parser.error("An explicit HTTP(S) MLflow server is required")

    source = load_modeling_pipeline(args.dataset, args.manifest, args.summary)
    if source.data.dataset_sha256 != DATASET_SHA256:
        raise ValueError("Dataset hash differs from frozen RF/GNN comparison")
    split = make_scaffold_calibration_split(
        source, seed=719, calibration_fraction=0.20, threshold_fraction=0.20,
        minimum_target_labels=10, minimum_dual_per_class=50, max_attempts=1,
    )
    fit_only = build_fit_only_tuning_pipeline(source, split, n_splits=5)
    if fit_only.stage_c_assignment_sha256 != FROZEN_STAGE_C_ASSIGNMENT_SHA256:
        raise ValueError("Frozen Stage C assignment mismatch")
    if len(fit_only.data.train) != 5364 or len(fit_only.folds) != 5:
        raise ValueError("Fit-only population or fold count differs from frozen study")
    bank = build_morgan_feature_bank(fit_only.data.train)
    storage = f"sqlite:///{args.optuna_db.resolve().as_posix()}"
    results = {}
    study_metadata = {}
    for target in ("egfr", "her2"):
        study = optuna.load_study(study_name=STUDIES[target], storage=storage)
        config = validate_study(study, target=target, fit_only=fit_only, bank=bank)
        print(f"Validated {target.upper()} best trial {study.best_trial.number} "
              f"and fingerprint/fold provenance")
        results[target] = (config, study)
        study_metadata[target] = {
            "name": study.study_name,
            "best_trial": study.best_trial.number,
            "best_cv_mean_mae": study.best_value,
            "config": asdict(config),
        }

    gnn_rows = read_gnn_oof_csv(args.gnn_oof)
    # Perform all identity/fold/mask checks after RF OOF generation.
    rf_egfr = run_rf_scaffold_cv(fit_only, bank, "egfr", results["egfr"][0])
    rf_her2 = run_rf_scaffold_cv(fit_only, bank, "her2", results["her2"][0])
    matched = align_rf_gnn_oof(fit_only, rf_egfr, rf_her2, gnn_rows)
    if len(matched) != 5364:
        raise ValueError("Matched OOF count differs from fit-only data")
    report = compare_aligned(matched)
    if (report["pooled"]["rf"]["egfr_n"],
        report["pooled"]["rf"]["her2_n"],
        report["pooled"]["rf"]["paired_n"]) != (4465, 1596, 697):
        raise ValueError("Matched label counts differ from frozen Stage C audit")

    report["provenance"] = {
        "dataset_sha256": source.data.dataset_sha256,
        "outer_manifest_sha256": source.data.manifest_sha256,
        "stage_c_assignment_sha256": fit_only.stage_c_assignment_sha256,
        "fold_assignments_sha256": compute_fold_manifest_sha256(fit_only),
        "fingerprint_sha256": bank.fingerprint_sha256,
        "gnn_oof_sha256": hashlib.sha256(args.gnn_oof.read_bytes()).hexdigest(),
        "optuna_studies": study_metadata,
        "rf_selectors_from_fit_only_cv": True,
        "cv_development_not_independent_post_selection_test": True,
        "calibration_scored": False,
        "threshold_scored": False,
        "development_validation_scored": False,
        "locked_test_scored": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = "M3_5a_RF_vs_GINE_matched_oof"
    paths = {
        "report": args.output_dir / f"{stem}.json",
        "molecules": args.output_dir / f"{stem}_molecules.csv",
        "scaffolds": args.output_dir / f"{stem}_scaffolds.csv",
    }
    write_aligned_csv(paths["molecules"], matched)
    write_csv(paths["scaffolds"], build_scaffold_report(matched))

    # Log to the same HTTP MLflow server used by M2 and M3, not a new local DB.
    import mlflow

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.mlflow_experiment)
    with mlflow.start_run(run_name="M3_5a_RF_vs_GINE_Matched_FitOnly_OOF") as run:
        report["mlflow_run_id"] = run.info.run_id
        mlflow.set_tags({
            "stage": "M3.5a", "comparison": "rf_vs_gine",
            "evaluation": "matched_fit_only_scaffold_oof",
            "dataset_sha256": source.data.dataset_sha256,
            "stage_c_assignment_sha256": fit_only.stage_c_assignment_sha256,
            "fold_assignments_sha256": compute_fold_manifest_sha256(fit_only),
            "calibration_scored": "false", "threshold_scored": "false",
            "development_validation_scored": "false", "locked_test_scored": "false",
        })
        for target, meta in study_metadata.items():
            mlflow.log_params({
                f"{target}_study_name": meta["name"],
                f"{target}_best_trial": meta["best_trial"],
                f"{target}_rf_n_estimators": meta["config"]["n_estimators"],
            })
        for method in ("rf", "gnn"):
            mlflow.log_metrics(numeric_metrics(report["pooled"][method], f"pooled_{method}_"))
        for entry in report["folds"]:
            for method in ("rf", "gnn"):
                mlflow.log_metrics(numeric_metrics(
                    entry[method], f"fold_{entry['fold']}_{method}_"
                ))
        paths["report"].write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        for path in paths.values():
            mlflow.log_artifact(str(path), artifact_path="evidence")
        print(f"MLflow comparison run: {run.info.run_id}")

    rf, gnn = report["pooled"]["rf"], report["pooled"]["gnn"]
    print(f"Pooled RF EGFR/HER2 MAE: {rf['egfr_mae']:.4f} / {rf['her2_mae']:.4f}")
    print(f"Pooled GINE EGFR/HER2 MAE: {gnn['egfr_mae']:.4f} / {gnn['her2_mae']:.4f}")
    print(f"Regression-derived dual AP: RF={rf['dual_min_prediction_ap']:.4f}, "
          f"GINE={gnn['dual_min_prediction_ap']:.4f}")
    for kind, path in paths.items():
        print(f"Saved {kind}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
