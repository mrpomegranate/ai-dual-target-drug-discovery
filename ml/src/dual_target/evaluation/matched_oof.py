"""Strict molecule-aligned comparison of fit-only RF and GNN OOF predictions.

The caller supplies the *existing* frozen fit-only scaffold pipeline, independent
EGFR/HER2 regression RFCVResult objects, and the GNN reference OOF CSV. No new
splits, model fitting, calibration, or reserved-label access occurs here.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

TARGETS = ("egfr", "her2")
METHODS = ("rf", "gnn")
LABEL_TOLERANCE = 1e-4  # Float32 GNN CSV vs original float64 M2 potency.
GNN_COLUMNS = {
    "structure_id", "fold", "egfr_mask", "her2_mask", "egfr_pic50",
    "her2_pic50", "egfr_pred", "her2_pred",
}


def _finite(value: Any, context: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Nonfinite {context}")
    return number


def read_gnn_oof_csv(path: str | Path) -> list[dict[str, Any]]:
    """Read the existing M3.5 CSV, parsing masks strictly and missing labels as None."""
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not GNN_COLUMNS.issubset(reader.fieldnames or []):
            raise ValueError("GNN OOF CSV is missing required columns")
        rows: list[dict[str, Any]] = []
        for raw in reader:
            sid = raw["structure_id"]
            if not sid:
                raise ValueError("GNN OOF contains an empty molecule ID")
            fold = int(raw["fold"])
            row: dict[str, Any] = {"structure_id": sid, "fold": fold}
            for target in TARGETS:
                mask_value = raw[f"{target}_mask"]
                if mask_value not in {"True", "False"}:
                    raise ValueError(f"Invalid {target} mask for {sid}")
                mask = mask_value == "True"
                observed = raw[f"{target}_pic50"]
                if mask and not observed:
                    raise ValueError(f"Missing measured {target} label for {sid}")
                if not mask and observed:
                    raise ValueError(f"Masked {target} label present for {sid}")
                row[f"{target}_mask"] = mask
                row[f"{target}_pic50"] = (
                    _finite(observed, f"{target} label") if mask else None
                )
                row[f"{target}_pred"] = _finite(
                    raw[f"{target}_pred"], f"GNN {target} prediction"
                )
            rows.append(row)
    if not rows:
        raise ValueError("GNN OOF CSV is empty")
    return rows


def _unique_by_id(rows: Sequence[Any], name: str) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for item in rows:
        sid = item.structure_id if hasattr(item, "structure_id") else item["structure_id"]
        if not sid or sid in mapping:
            raise ValueError(f"Duplicate/empty molecule ID in {name}: {sid}")
        mapping[sid] = item
    return mapping


def align_rf_gnn_oof(
    fit_only: Any,
    rf_egfr: Any,
    rf_her2: Any,
    gnn_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Require exact frozen molecule/fold alignment; never use a permissive inner join."""
    if (rf_egfr.task, rf_her2.task) != TARGETS:
        raise ValueError("Expected EGFR and HER2 regression RFCVResults")

    records = tuple(fit_only.data.train)
    source = _unique_by_id(records, "frozen fitting records")
    gnn = _unique_by_id(gnn_rows, "GNN OOF")
    if set(gnn) != set(source):
        raise ValueError("GNN OOF identities differ from frozen fitting records")

    folds: dict[str, int] = {}
    for fold in fit_only.folds:
        for record in fold.holdout:
            if record.structure_id in folds:
                raise ValueError("Repeated molecule in frozen CV holdouts")
            folds[record.structure_id] = fold.number
    if set(folds) != set(source):
        raise ValueError("Frozen CV holdouts do not cover fit molecules exactly")

    rf_maps: dict[str, dict[str, Any]] = {}
    for target, result in zip(TARGETS, (rf_egfr, rf_her2), strict=True):
        mapping = _unique_by_id(result.oof_predictions, f"RF {target} OOF")
        eligible = {r.structure_id for r in records if getattr(r, f"{target}_mask") == 1}
        if set(mapping) != eligible:
            raise ValueError(f"RF {target} OOF does not cover measured targets exactly")
        rf_maps[target] = mapping

    aligned = []
    for record in records:  # Preserve frozen source ordering.
        sid = record.structure_id
        gnn_row = gnn[sid]
        fold = folds[sid]
        if type(gnn_row["fold"]) is not int or gnn_row["fold"] != fold:
            raise ValueError(f"GNN fold mismatch for {sid}")
        output = {"structure_id": sid, "fold": fold,
                  "scaffold_group_id": record.scaffold_group_id}
        for target in TARGETS:
            observed_mask = bool(getattr(record, f"{target}_mask"))
            if type(gnn_row[f"{target}_mask"]) is not bool or (
                gnn_row[f"{target}_mask"] != observed_mask
            ):
                raise ValueError(f"GNN {target} mask mismatch for {sid}")
            label = getattr(record, f"{target}_pic50")
            gnn_label = gnn_row[f"{target}_pic50"]
            gnn_prediction = _finite(gnn_row[f"{target}_pred"], f"GNN {target} prediction")
            output[f"{target}_mask"] = observed_mask
            output[f"{target}_pic50"] = label
            output[f"{target}_gnn_pred"] = gnn_prediction
            if observed_mask:
                true_label = _finite(label, f"frozen {target} label")
                if gnn_label is None or not math.isclose(
                    true_label, _finite(gnn_label, f"GNN {target} label"),
                    rel_tol=0.0, abs_tol=LABEL_TOLERANCE,
                ):
                    raise ValueError(f"GNN {target} label mismatch for {sid}")
                rf_row = rf_maps[target][sid]
                if rf_row.fold != fold or rf_row.scaffold_group_id != record.scaffold_group_id:
                    raise ValueError(f"RF {target} fold/scaffold mismatch for {sid}")
                if not math.isclose(
                    true_label, _finite(rf_row.observed, f"RF {target} label"),
                    rel_tol=0.0, abs_tol=LABEL_TOLERANCE,
                ):
                    raise ValueError(f"RF {target} label mismatch for {sid}")
                output[f"{target}_rf_pred"] = _finite(
                    rf_row.predicted, f"RF {target} prediction"
                )
            else:
                if label is not None or gnn_label is not None:
                    raise ValueError(f"Masked {target} label for {sid}")
                output[f"{target}_rf_pred"] = None
        aligned.append(output)
    return aligned


def score_aligned(rows: Sequence[Mapping[str, Any]], method: str) -> dict[str, Any]:
    """Score one model with identical target masks and paired dual examples."""
    if method not in METHODS or not rows:
        raise ValueError("Require method rf/gnn and nonempty matched records")
    result: dict[str, Any] = {"molecules": len(rows)}
    for target in TARGETS:
        measured = [row for row in rows if row[f"{target}_mask"]]
        if not measured:
            raise ValueError(f"No measured {target} labels")
        observed = np.asarray([r[f"{target}_pic50"] for r in measured], dtype=float)
        prediction = np.asarray([r[f"{target}_{method}_pred"] for r in measured], dtype=float)
        if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(prediction)):
            raise ValueError(f"Nonfinite measured {target} values")
        result[f"{target}_n"] = len(measured)
        result[f"{target}_mae"] = float(np.mean(np.abs(observed - prediction)))
        result[f"{target}_spearman"] = (
            float(spearmanr(observed, prediction).statistic)
            if len(measured) > 1 and np.ptp(observed) > 0 and np.ptp(prediction) > 0
            else None
        )
    paired = [r for r in rows if r["egfr_mask"] and r["her2_mask"]]
    labels = np.asarray([
        int(r["egfr_pic50"] > 6.0 and r["her2_pic50"] > 6.0)
        for r in paired
    ], dtype=int)
    result["paired_n"] = len(paired)
    result["dual_active_n"] = int(labels.sum())
    result["dual_inactive_n"] = int(len(labels) - labels.sum())
    result["dual_prevalence"] = float(labels.mean()) if len(labels) else None
    result["dual_min_prediction_ap"] = (
        float(average_precision_score(labels, [
            min(row[f"egfr_{method}_pred"], row[f"her2_{method}_pred"])
            for row in paired
        ]))
        if 0 < labels.sum() < len(labels) else None
    )
    result["balanced_mae"] = (result["egfr_mae"] + result["her2_mae"]) / 2
    return result


def compare_aligned(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize pooled + per-fold matched metrics; signed deltas are GNN minus RF."""
    if not rows:
        raise ValueError("No matched OOF rows")
    groups = {int(row["fold"]) for row in rows}
    def metrics(subset: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        rf, gnn = score_aligned(subset, "rf"), score_aligned(subset, "gnn")
        names = ("egfr_mae", "her2_mae", "balanced_mae", "dual_min_prediction_ap")
        differences = {
            name: (gnn[name] - rf[name])
            if gnn[name] is not None and rf[name] is not None else None
            for name in names
        }
        return {"rf": rf, "gnn": gnn, "gnn_minus_rf": differences}
    return {
        "purpose": "matched_fit_only_rf_vs_gine_oof_no_reserved_labels",
        "pooled": metrics(rows),
        "folds": [
            {"fold": number, **metrics([r for r in rows if int(r["fold"]) == number])}
            for number in sorted(groups)
        ],
    }


def write_aligned_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write auditable per-molecule comparison including restored scaffold IDs."""
    if not rows:
        raise ValueError("Cannot export empty comparison")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
