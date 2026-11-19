"""Report difficult scaffolds from fit-only OOF predictions, never reserved labels.

This does NOT calibrate any model. Formal conformal calibration requires
predictions from models freshly trained on split.fit and scored on the distinct
split.calibration role. This script is descriptive model-development evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dual_target.evaluation.scaffold_diagnostics import (
    MODEL_NAMES,
    TARGETS,
    rank_difficult_scaffolds,
    scaffold_error_table,
    summarize_scaffold_diagnostics,
    validate_fit_only_oof,
)

DEFAULT_OOF = Path("reports/milestones/M3_6d_three_model_and_ensemble_fit_only_OOF.csv")
OUTPUT_STEM = "M3_7a_fit_only_scaffold_difficulty"
EXPERIMENT = "EGFR_HER2_M2_Scaffold_Benchmark"


def plot_difficult(groups: pd.DataFrame, target: str, path: Path, *, rank_by: str) -> None:
    """Top-scaffold MAE with molecule counts, labeled as exploratory."""
    top = groups.loc[groups.target == target].copy().iloc[::-1]
    if top.empty:
        raise ValueError("Cannot plot an empty scaffold ranking")
    fig, ax = plt.subplots(figsize=(11, max(4.5, 0.5 * len(top) + 2)))
    y = np.arange(len(top))
    offsets = [-0.27, -0.09, 0.09, 0.27]
    keys = ["rf", "gine", "gatv2", "rf75_gine25"]
    names = ["Morgan RF", "GINE", "GATv2", "75% RF + 25% GINE"]
    for offset, key, name in zip(offsets, keys, names, strict=True):
        ax.barh(y + offset, top[f"{key}_mae"], height=.17, label=name)
    ax.set_yticks(y, [f"{x[-12:]} (n={n})" for x, n in zip(
        top.scaffold_group_id, top.n, strict=True
    )])
    ax.set_xlabel("Scaffold-held-out pIC50 MAE (lower is better)")
    ax.set_title(f"{target.upper()}: difficult scaffolds ranked by {rank_by} (fit-only OOF)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.2)
    fig.text(.04, .01, "Scaffold IDs truncated for display; refer to complete CSV. No test labels used.",
             fontsize=8)
    fig.tight_layout(rect=(0, .025, 1, 1))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", type=Path, default=DEFAULT_OOF)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/milestones"))
    parser.add_argument("--minimum-n", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--rank-by", choices=[f"{m}_mae" for m in MODEL_NAMES],
                        default="rf_mae")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--mlflow-uri", default="http://127.0.0.1:5000")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args(argv)

    if not args.oof.is_file():
        parser.error(f"Fit-only OOF missing: {args.oof}")
    frame = pd.read_csv(args.oof)
    validate_fit_only_oof(frame, expected_n=5364)
    groups = scaffold_error_table(frame, minimum_n=args.minimum_n)
    ranked = rank_difficult_scaffolds(groups, top_k=args.top_k, rank_by=args.rank_by)
    summary = summarize_scaffold_diagnostics(frame, groups)
    summary["input_oof_sha256"] = hashlib.sha256(args.oof.read_bytes()).hexdigest()
    summary["ranking"] = {"by": args.rank_by, "minimum_n": args.minimum_n, "top_k": args.top_k}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "all_scaffolds": args.output_dir / f"{OUTPUT_STEM}_all.csv",
        "ranking": args.output_dir / f"{OUTPUT_STEM}_top.csv",
        "summary": args.output_dir / f"{OUTPUT_STEM}.json",
        "egfr_plot": args.output_dir / f"{OUTPUT_STEM}_egfr.png",
        "her2_plot": args.output_dir / f"{OUTPUT_STEM}_her2.png",
    }
    existing = [str(p) for p in paths.values() if p.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Output files exist; pass --overwrite to replace: {existing}")
    groups.to_csv(paths["all_scaffolds"], index=False)
    ranked.to_csv(paths["ranking"], index=False)
    for target in TARGETS:
        plot_difficult(ranked, target, paths[f"{target}_plot"], rank_by=args.rank_by)

    if not args.no_mlflow:
        import mlflow
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(EXPERIMENT)
        with mlflow.start_run(run_name="M3_7a_FitOnly_Scaffold_Difficulty") as run:
            summary["mlflow_run_id"] = run.info.run_id
            mlflow.set_tags({
                "stage": "M3.7a",
                "evaluation": "fit_only_oof_scaffold_error_audit",
                "calibration_scored": "false", "threshold_scored": "false",
                "development_validation_scored": "false", "locked_test_scored": "false",
            })
            mlflow.log_params({"minimum_n": args.minimum_n, "top_k": args.top_k,
                               "rank_by": args.rank_by})
            mlflow.log_metrics({
                f"{t}_{model}_oof_mae": summary["targets"][t]["pooled_mae"][model]
                for t in TARGETS for model in MODEL_NAMES
            })
            paths["summary"].write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            for path in paths.values():
                mlflow.log_artifact(str(path), artifact_path="scaffold_diagnostics")
            print(f"MLflow run: {run.info.run_id}")
    else:
        paths["summary"].write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    for target in TARGETS:
        counts = summary["targets"][target]
        print(f"{target.upper()}: {counts['scaffold_groups_with_labels']} total scaffolds, "
              f"{counts['reportable_scaffolds']} have >= {args.minimum_n} measured labels")
    for name, path in paths.items():
        print(f"Saved {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
