"""Select a training-only, scaffold-disjoint Stage C split by fixed balance rules.

This is label-aware GROUP STRATIFICATION, not model-performance tuning.
We scan seeds in ascending order and accept the FIRST split that meets
predeclared training-only thresholds. No development-validation labels, final
holdout labels, predictions, model metrics, or calibration scores are consulted.

The selected seed must then be passed to audit_scaffold_calibration.py with
--max-attempts 1 to create a new, versioned assignment manifest and MLflow run.
The original audit is retained as evidence; do not overwrite it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from dual_target.calibration.scaffold_calibration import (
    audit_calibration_records,
    make_scaffold_calibration_split,
)

DEFAULT_DATASET = Path("data/interim/M2_1_scaffold_split_molecules.csv")
DEFAULT_MANIFEST = Path("data/interim/M2_1_scaffold_split_manifest.csv")
DEFAULT_SUMMARY = Path("reports/audits/M2_1_scaffold_split_summary.json")
DEFAULT_OUTPUT_DIR = Path("reports/calibration")
DEFAULT_MLFLOW_DB = Path("reports/mlflow/mlflow.db")
STAGE_B_RUN_ID = "12c9fc400b354f6da4e993e69e36b1cb"
PREVIOUS_SPLIT_RUN_ID = "2e8b155d2ebd4cf3a2ca6ffc71a947c3"


@dataclass(frozen=True)
class SelectionRules:
    start_seed: int = 714
    candidates: int = 200
    max_prevalence_deviation: float = 0.05
    min_paired_per_reserved_role: int = 150
    min_class_per_reserved_role: int = 50
    calibration_fraction: float = 0.20
    threshold_fraction: float = 0.20

    def validate(self) -> None:
        if self.start_seed < 0 or self.candidates < 1:
            raise ValueError("start_seed must be >= 0 and candidates >= 1")
        if not 0 < self.max_prevalence_deviation < 1:
            raise ValueError("max_prevalence_deviation must be between 0 and 1")
        if self.min_paired_per_reserved_role < 2:
            raise ValueError("min_paired_per_reserved_role must be >= 2")
        if self.min_class_per_reserved_role < 1:
            raise ValueError("min_class_per_reserved_role must be >= 1")
        if not (0 < self.calibration_fraction < 1 and 0 < self.threshold_fraction < 1):
            raise ValueError("fractions must be between 0 and 1")
        if self.calibration_fraction + self.threshold_fraction >= 1:
            raise ValueError("fractions must sum to less than one")


def _prevalence(audit) -> float:
    if audit.paired_labels == 0:
        raise ValueError("No accepted paired labels to assess class balance")
    return audit.dual_active / audit.paired_labels


def select_first_balanced_split(pipeline, rules: SelectionRules):
    """Return first feasible split; no performance-based seed optimization."""
    rules.validate()
    reference = audit_calibration_records(tuple(pipeline.data.train))
    reference_prev = _prevalence(reference)
    candidates_checked: list[dict] = []

    for seed in range(rules.start_seed, rules.start_seed + rules.candidates):
        try:
            split = make_scaffold_calibration_split(
                pipeline,
                calibration_fraction=rules.calibration_fraction,
                threshold_fraction=rules.threshold_fraction,
                seed=seed,
                minimum_target_labels=10,
                minimum_dual_per_class=1,
                max_attempts=1,  # crucial: never silently change this seed
            )
        except ValueError as exc:
            if not str(exc).startswith("No valid three-way scaffold split"):
                raise
            candidates_checked.append({"seed": seed, "status": "insufficient_labels"})
            continue

        rates = {
            role: _prevalence(getattr(split, f"{role}_audit"))
            for role in ("fit", "calibration", "threshold")
        }
        reserved = (split.calibration_audit, split.threshold_audit)
        enough = all(
            audit.paired_labels >= rules.min_paired_per_reserved_role
            and audit.dual_active >= rules.min_class_per_reserved_role
            and audit.not_dual_active >= rules.min_class_per_reserved_role
            for audit in reserved
        )
        balanced = all(
            abs(rate - reference_prev) <= rules.max_prevalence_deviation
            for rate in rates.values()
        )
        candidate = {
            "seed": seed,
            "status": "accepted" if enough and balanced else "rejected",
            "paired_prevalence": rates,
            "calibration_paired": split.calibration_audit.paired_labels,
            "calibration_negative": split.calibration_audit.not_dual_active,
            "threshold_paired": split.threshold_audit.paired_labels,
            "threshold_negative": split.threshold_audit.not_dual_active,
            "assignment_sha256": split.assignment_sha256,
        }
        candidates_checked.append(candidate)
        if enough and balanced:
            return split, reference, candidates_checked

    raise ValueError(
        "No split passed the predeclared class-balance criteria after "
        f"{rules.candidates} seeds. Do not silently relax rules; review "
        "their feasibility and version any changes."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-seed", type=int, default=714)
    parser.add_argument("--candidates", type=int, default=200)
    parser.add_argument("--max-prevalence-deviation", type=float, default=0.05)
    parser.add_argument("--min-paired", type=int, default=150)
    parser.add_argument("--min-class", type=int, default=50)
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    parser.add_argument("--threshold-fraction", type=float, default=0.20)
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--experiment", default="EGFR_HER2_M2_Scaffold_Benchmark")
    parser.add_argument("--run-name", default="M2_2_StageC_TrainingOnly_SplitBalance")
    return parser.parse_args()


def save_evidence(payload: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    digest = hashlib.sha256(content).hexdigest()[:12]
    path = output_dir / f"M2_2_C3_seed_selection_{digest}.json"
    if path.exists() and path.read_bytes() != content:
        raise FileExistsError(f"Refusing overwrite: {path}")
    path.write_bytes(content)
    return path


def main() -> None:
    from dual_target.data.pipeline import load_modeling_pipeline

    args = parse_args()
    rules = SelectionRules(
        start_seed=args.start_seed,
        candidates=args.candidates,
        max_prevalence_deviation=args.max_prevalence_deviation,
        min_paired_per_reserved_role=args.min_paired,
        min_class_per_reserved_role=args.min_class,
        calibration_fraction=args.calibration_fraction,
        threshold_fraction=args.threshold_fraction,
    )
    pipeline = load_modeling_pipeline(
        args.dataset, args.manifest, args.summary, n_splits=5
    )
    split, reference, checked = select_first_balanced_split(pipeline, rules)
    payload = {
        "method": "first_seed_meeting_predeclared_training_only_balance_rules",
        "rules": asdict(rules),
        "dataset_sha256": split.dataset_sha256,
        "frozen_manifest_sha256": split.manifest_sha256,
        "previous_stage_c3_run_id": PREVIOUS_SPLIT_RUN_ID,
        "stage_b_run_id": STAGE_B_RUN_ID,
        "training_paired_prevalence": _prevalence(reference),
        "selected_seed": split.seed_used,
        "selected_assignment_sha256": split.assignment_sha256,
        "selected_audits": {
            role: asdict(getattr(split, f"{role}_audit"))
            for role in ("fit", "calibration", "threshold")
        },
        "candidates_checked": checked,
        "no_model_metric_or_validation_label_used": True,
        "test_partition_not_loaded": True,
    }
    artifact_path = save_evidence(payload, args.output_dir)

    import mlflow

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    elif not os.environ.get("MLFLOW_TRACKING_URI"):
        uri = DEFAULT_MLFLOW_DB.resolve().as_posix()
        DEFAULT_MLFLOW_DB.parent.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(f"sqlite:///{uri}")
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.set_tags({
            "milestone": "M2.2_stage_C3_split_balance",
            "selection_basis": "training_label_prevalence_only",
            "no_model_performance_selection": "true",
            "test_labels_used": "false",
            "stage_b_parent_run_id": STAGE_B_RUN_ID,
            "previous_stage_c3_run_id": PREVIOUS_SPLIT_RUN_ID,
        })
        mlflow.log_params({
            **asdict(rules),
            "dataset_sha256": split.dataset_sha256,
            "manifest_sha256": split.manifest_sha256,
            "assignment_sha256": split.assignment_sha256,
            "selected_seed": split.seed_used,
        })
        mlflow.log_metrics({
            "training_dual_prevalence": _prevalence(reference),
            "fit_dual_prevalence": _prevalence(split.fit_audit),
            "calibration_dual_prevalence": _prevalence(split.calibration_audit),
            "threshold_dual_prevalence": _prevalence(split.threshold_audit),
            "seeds_examined": len(checked),
        })
        mlflow.log_artifact(str(artifact_path), artifact_path="selection")
        run_id = run.info.run_id

    print("\nSelected Stage C scaffold-balanced assignment:")
    print(f"Seed: {split.seed_used} (first passing among {len(checked)} examined)")
    print(f"Training prevalence: {_prevalence(reference):.3f}")
    for role in ("fit", "calibration", "threshold"):
        audit = getattr(split, f"{role}_audit")
        print(
            f"{role:12s} molecules={audit.molecules:4d} "
            f"paired={audit.paired_labels:3d} dual+={audit.dual_active:3d} "
            f"dual-={audit.not_dual_active:3d} "
            f"prevalence={_prevalence(audit):.3f}"
        )
    print(f"Assignment SHA-256: {split.assignment_sha256}")
    print(f"Selection evidence: {artifact_path}")
    print(f"MLflow selection run: {run_id}")
    print("\nRe-run the existing audit with the selected seed and --max-attempts 1.")


if __name__ == "__main__":
    main()
