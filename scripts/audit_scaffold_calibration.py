"""Stage C.3: audit a three-way scaffold split of frozen training molecules.

Only M2.1 training records are assigned to fit/calibration/threshold roles.
Development validation is used by the underlying split module only to verify
molecule/scaffold disjointness; test potency labels are never loaded.

This is a split audit, NOT a fitted/calibrated predictive model. Store its
content-addressed manifest and counts in reports/calibration and MLflow so
later calibration runs can reference exactly the same training assignment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
from dataclasses import asdict, fields
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from dual_target.calibration.scaffold_calibration import (
    CalibrationAudit,
    audit_calibration_records,
    make_scaffold_calibration_split,
)
from dual_target.data.modeling_dataset import sha256_file
from dual_target.data.pipeline import load_modeling_pipeline

DEFAULT_DATASET = Path("data/interim/M2_1_scaffold_split_molecules.csv")
DEFAULT_MANIFEST = Path("data/interim/M2_1_scaffold_split_manifest.csv")
DEFAULT_SUMMARY = Path("reports/audits/M2_1_scaffold_split_summary.json")
DEFAULT_MLFLOW_DB = Path("reports/mlflow/mlflow.db")
DEFAULT_REPORT_DIR = Path("reports/calibration")
DEFAULT_STAGE_B_RUN_ID = "12c9fc400b354f6da4e993e69e36b1cb"
ROLES = ("fit", "calibration", "threshold")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    parser.add_argument("--threshold-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=714)
    parser.add_argument("--min-target-labels", type=int, default=10)
    parser.add_argument("--min-dual-per-class", type=int, default=20)
    parser.add_argument("--max-attempts", type=int, default=50)
    parser.add_argument("--experiment", default="EGFR_HER2_M2_Scaffold_Benchmark")
    parser.add_argument("--run-name", default="M2_2_StageC_Scaffold_Split_Audit")
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--stage-b-run-id", default=DEFAULT_STAGE_B_RUN_ID)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.min_target_labels < 1 or args.min_dual_per_class < 1:
        parser.error("Minimum label counts must be positive")
    if args.max_attempts < 1:
        parser.error("--max-attempts must be positive")
    if not (0 < args.calibration_fraction < 1 and 0 < args.threshold_fraction < 1):
        parser.error("Calibration and threshold fractions must be between 0 and 1")
    if args.calibration_fraction + args.threshold_fraction >= 1:
        parser.error("Calibration and threshold fractions must sum to less than 1")
    return args


def sqlite_uri(path: Path) -> str:
    full_path = path.resolve()
    full_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{full_path.as_posix()}"


def configure_mlflow(args: argparse.Namespace) -> None:
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    elif not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(sqlite_uri(DEFAULT_MLFLOW_DB))
    mlflow.set_experiment(args.experiment)


def git_state() -> tuple[str, str]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True,
            check=True,
        ).stdout.strip()
        return commit, str(bool(dirty))
    except (OSError, subprocess.CalledProcessError):
        return "unknown", "unknown"


def audit_rows(split, original: CalibrationAudit) -> list[dict[str, int | float | str]]:
    """Build role and total rows; fail if label accounting changes."""
    subsets = {
        "fit": split.fit_audit,
        "calibration": split.calibration_audit,
        "threshold": split.threshold_audit,
    }
    for field in fields(CalibrationAudit):
        if sum(getattr(audit, field.name) for audit in subsets.values()) != getattr(
            original, field.name
        ):
            raise ValueError(f"Stage C label-count mismatch: {field.name}")

    results = []
    for role, audit in (*subsets.items(), ("total_training", original)):
        results.append({
            "role": role,
            **asdict(audit),
            "dual_prevalence": (
                audit.dual_active / audit.paired_labels if audit.paired_labels else 0.0
            ),
            "molecule_fraction": audit.molecules / original.molecules,
            "scaffold_fraction": audit.scaffolds / original.scaffolds,
        })
    return results


def make_summary(pipeline, split, args: argparse.Namespace) -> dict:
    original = audit_calibration_records(tuple(pipeline.data.train))
    rows = audit_rows(split, original)
    return {
        "stage": "M2_2_StageC_split_audit",
        "source": {
            "dataset_sha256": split.dataset_sha256,
            "manifest_sha256": split.manifest_sha256,
            "curation_summary_sha256": sha256_file(args.summary),
            "development_validation_molecules": len(pipeline.data.validation),
            "test_labels_used": False,
        },
        "split": {
            "assignment_sha256": split.assignment_sha256,
            "requested_seed": args.seed,
            "seed_used": split.seed_used,
            "requested_calibration_fraction_of_scaffolds": (
                split.requested_calibration_fraction
            ),
            "requested_threshold_fraction_of_scaffolds": (
                split.requested_threshold_fraction
            ),
            "min_target_labels": args.min_target_labels,
            "min_dual_per_class": args.min_dual_per_class,
            "max_attempts": args.max_attempts,
            "roles": list(ROLES),
            "activity_rule": "EGFR pIC50 > 6 AND HER2 pIC50 > 6",
            "fraction_unit": "scaffold groups, not molecules",
        },
        "audits": rows,
        "caveats": [
            (
                "Optuna previously used training records for model selection;"
                " these role assignments are not independent of that selection."
            ),
            (
                "No model was fitted and no probability calibrator, conformal interval,"
                " or decision threshold was estimated in this run."
            ),
            (
                "Valid conformal coverage on novel scaffolds is not guaranteed by"
                " this split, particularly under distribution shift."
            ),
        ],
    }


def _persist_if_unchanged(path: Path, content: bytes) -> None:
    """Never silently replace an existing artifact of the same identity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(
                f"Existing evidence differs; refusing overwrite: {path}"
            )
        return
    path.write_bytes(content)


def write_artifacts(split, summary: dict, output_dir: Path) -> tuple[Path, Path]:
    prefix = f"M2_2_C3_scaffold_split_{split.assignment_sha256[:12]}"
    manifest_path = output_dir / f"{prefix}.csv"
    summary_path = output_dir / f"{prefix}.json"

    out = io.StringIO(newline="")
    writer = csv.DictWriter(
        out, fieldnames=("structure_id", "scaffold_group_id", "role"),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(split.manifest_rows())
    manifest_bytes = out.getvalue().encode("utf-8")
    summary["split"]["manifest_csv_sha256"] = hashlib.sha256(
        manifest_bytes
    ).hexdigest()
    summary_bytes = (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )

    _persist_if_unchanged(manifest_path, manifest_bytes)
    _persist_if_unchanged(summary_path, summary_bytes)
    return manifest_path, summary_path


def log_split_figure(rows: list[dict]) -> None:
    roles = [row for row in rows if row["role"] in ROLES]
    labels = [str(row["role"]).capitalize() for row in roles]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    try:
        axes[0].bar(labels, [int(row["molecules"]) for row in roles])
        axes[0].set(title="Stage C training-only scaffold split", ylabel="Molecules")
        axes[1].bar(
            labels, [int(row["dual_active"]) for row in roles], label="Dual-active"
        )
        axes[1].bar(
            labels,
            [int(row["not_dual_active"]) for row in roles],
            bottom=[int(row["dual_active"]) for row in roles],
            label="Not dual-active",
        )
        axes[1].set(title="Accepted paired labels by role", ylabel="Paired molecules")
        axes[1].legend()
        fig.tight_layout()
        mlflow.log_figure(fig, "plots/stage_c_split_audit.png")
    finally:
        plt.close(fig)


def log_mlflow(pipeline, split, summary: dict, files: tuple[Path, Path], args) -> str:
    configure_mlflow(args)
    with mlflow.start_run(run_name=args.run_name) as run:
        commit, dirty = git_state()
        mlflow.set_tags({
            "milestone": "M2.2_stage_C3",
            "analysis": "training_only_split_audit",
            "calibration_fitted": "false",
            "threshold_tuned": "false",
            "test_labels_used": "false",
            "development_validation_used_for_split_selection": "false",
            "stage_b_parent_run_id": args.stage_b_run_id,
            "git_commit": commit,
            "git_dirty": dirty,
        })
        mlflow.log_params({
            "dataset_sha256": split.dataset_sha256,
            "manifest_sha256": split.manifest_sha256,
            "curation_summary_sha256": summary["source"]["curation_summary_sha256"],
            "assignment_sha256": split.assignment_sha256,
            "assignment_csv_sha256": summary["split"]["manifest_csv_sha256"],
            "requested_seed": args.seed,
            "seed_used": split.seed_used,
            "calibration_scaffold_fraction": args.calibration_fraction,
            "threshold_scaffold_fraction": args.threshold_fraction,
            "min_target_labels": args.min_target_labels,
            "min_dual_per_class": args.min_dual_per_class,
            "max_attempts": args.max_attempts,
            "original_training_molecules": len(pipeline.data.train),
            "dev_validation_molecules_excluded": len(pipeline.data.validation),
        })
        metrics = {}
        for row in summary["audits"]:
            if row["role"] not in ROLES:
                continue
            role = row["role"]
            for field in (
                "molecules", "scaffolds", "egfr_labels", "her2_labels",
                "paired_labels", "dual_active", "not_dual_active",
                "dual_prevalence", "molecule_fraction", "scaffold_fraction",
            ):
                metrics[f"{role}_{field}"] = float(row[field])
        mlflow.log_metrics(metrics)

        # Register exact assignments in MLflow, without any potency values.
        frame = pd.DataFrame(split.manifest_rows())
        input_dataset = mlflow.data.from_pandas(
            frame,
            name=f"M2_2_C3_roles_{split.assignment_sha256[:12]}",
            source=args.dataset.resolve().as_uri(),
        )
        mlflow.log_input(input_dataset, context="training_scaffold_roles")
        mlflow.log_artifact(str(files[0]), artifact_path="split")
        mlflow.log_artifact(str(files[1]), artifact_path="split")
        mlflow.log_artifact(str(args.manifest), artifact_path="provenance")
        mlflow.log_artifact(str(args.summary), artifact_path="provenance")
        mlflow.log_dict({
            "stage_b_parent_run_id": args.stage_b_run_id,
            "assignment_sha256": split.assignment_sha256,
            "dataset_sha256": split.dataset_sha256,
            "manifest_sha256": split.manifest_sha256,
            "calibration_models_trained": False,
            "calibration_estimated": False,
            "decision_threshold_selected": False,
            "final_test_labels_loaded": False,
        }, "provenance/protocol.json")
        log_split_figure(summary["audits"])
        return run.info.run_id


def print_audit(summary: dict, manifest_path: Path, summary_path: Path) -> None:
    print("\nSTAGE C.3 — SCAFFOLD-DISJOINT CALIBRATION SPLIT")
    print("All assignments use the original M2.1 TRAIN partition only.\n")
    print(
        f"{'Role':<17} {'Molecules':>9} {'Scaffolds':>10} {'EGFR':>7} "
        f"{'HER2':>7} {'Paired':>7} {'Dual+':>7} {'Dual-':>7} {'Prev.':>7}"
    )
    for row in summary["audits"]:
        print(
            f"{row['role']:<17} {row['molecules']:>9} {row['scaffolds']:>10} "
            f"{row['egfr_labels']:>7} {row['her2_labels']:>7} "
            f"{row['paired_labels']:>7} {row['dual_active']:>7} "
            f"{row['not_dual_active']:>7} {row['dual_prevalence']:>7.3f}"
        )
    print(f"\nAssignment SHA-256: {summary['split']['assignment_sha256']}")
    print(f"Seed used: {summary['split']['seed_used']}")
    print(f"Split manifest: {manifest_path}")
    print(f"Split audit: {summary_path}")
    for row in summary["audits"]:
        if row["role"] in ROLES and (
            row["dual_active"] < 30 or row["not_dual_active"] < 30
        ):
            print(
                f"CAUTION: {row['role']} has fewer than 30 examples in "
                "at least one dual class; evaluate calibration stability."
            )


def main() -> None:
    args = parse_args()
    pipeline = load_modeling_pipeline(
        args.dataset, args.manifest, args.summary, n_splits=5
    )
    split = make_scaffold_calibration_split(
        pipeline,
        calibration_fraction=args.calibration_fraction,
        threshold_fraction=args.threshold_fraction,
        seed=args.seed,
        minimum_target_labels=args.min_target_labels,
        minimum_dual_per_class=args.min_dual_per_class,
        max_attempts=args.max_attempts,
    )
    summary = make_summary(pipeline, split, args)
    files = write_artifacts(split, summary, args.output_dir)
    print_audit(summary, *files)
    run_id = log_mlflow(pipeline, split, summary, files, args)
    print(f"\nStage C.3 parent MLflow run: {run_id}")
    print(
        "Only assignments/audits were logged; "
        "no calibration or threshold was fitted."
    )


if __name__ == "__main__":
    main()
